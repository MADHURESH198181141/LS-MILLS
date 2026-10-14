import os
import ssl
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from insightface.app import FaceAnalysis
import time
import warnings
import requests
import urllib3
import pickle

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================
# FIX SSL
# ==============================
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# ==============================
# CONFIGURATION
# ==============================

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_METRIC_DIR   = os.path.dirname(_SCRIPT_DIR)
_PARENT_DIR   = os.path.dirname(_METRIC_DIR)

_candidate_images = os.path.join(_PARENT_DIR, "FGNET (1)", "FGNET", "images")
DATASET_PATH  = _candidate_images if os.path.exists(_candidate_images) else os.path.join(_METRIC_DIR, "FGNET", "images")
RESULT_PATH   = _METRIC_DIR
MODEL_SAVE    = os.path.join(_METRIC_DIR, "hybrid_model")

os.makedirs(RESULT_PATH,  exist_ok=True)
os.makedirs(MODEL_SAVE,   exist_ok=True)

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_AGE_GAP = 5
TRAIN_RATIO = 0.8    # 80% train / 20% test
EPOCHS      = 50
LR          = 5e-4
BATCH_SIZE  = 32

print(f"Using device : {DEVICE}")
print(f"Train/Test   : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")

# ==============================
# MODELS TO USE FOR HYBRID
# buffalo_l  → high accuracy
# buffalo_m  → balanced + fast
# ==============================

MODEL_A = "buffalo_l"   # Best accuracy
MODEL_B = "buffalo_m"   # Best lightweight

# ==============================
# PARSE FGNET FILENAMES
# 001A02.jpg → ID=001, AGE=02
# ==============================

def parse_filename(filename):
    try:
        base      = os.path.basename(filename)
        person_id = base[:3]
        age_str   = base.split("A")[1].split(".")[0]
        return person_id, int(age_str)
    except Exception:
        return None, None

# ==============================
# LOAD INSIGHTFACE MODEL
# ==============================

def load_insightface(model_name):
    try:
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if DEVICE == 'cuda'
            else ['CPUExecutionProvider']
        )
        app = FaceAnalysis(name=model_name, providers=providers)
        app.prepare(ctx_id=0 if DEVICE == 'cuda' else -1)
        print(f"  ✅ Loaded: {model_name}")
        return app
    except Exception as e:
        print(f"  ❌ Failed to load {model_name}: {e}")
        return None

# ==============================
# EXTRACT EMBEDDINGS FROM MODEL
# ==============================

def extract_embeddings(app, model_name):
    cache_path = os.path.join(MODEL_SAVE, f"embeddings_{model_name}.pkl")

    # Load from cache if exists
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings for {model_name}...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded: {len(data['embeddings'])} embeddings")
        return data['embeddings'], data['labels'], data['ages']

    embeddings = []
    labels     = []
    ages       = []
    failed     = 0

    image_files = [
        f for f in os.listdir(DATASET_PATH)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    for file in tqdm(image_files, desc=f"  Extracting [{model_name}]"):
        path = os.path.join(DATASET_PATH, file)
        try:
            img = cv2.imread(path)
            if img is None:
                failed += 1
                continue

            pid, age = parse_filename(file)
            if pid is None:
                failed += 1
                continue

            faces = app.get(img)
            if len(faces) == 0:
                failed += 1
                continue

            emb = faces[0].embedding
            emb = normalize(emb.reshape(1, -1))[0]

            embeddings.append(emb)
            labels.append(pid)
            ages.append(age)

        except Exception:
            failed += 1
            continue

    print(f"  Extracted: {len(embeddings)} | Failed: {failed}")

    # Cache embeddings to disk
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'embeddings': np.array(embeddings),
            'labels'    : np.array(labels),
            'ages'      : np.array(ages)
        }, f)

    return np.array(embeddings), np.array(labels), np.array(ages)

# ==============================
# BUILD PAIR DATASET
# Each sample = [emb_a, emb_b, label]
# label=1 genuine, label=0 impostor
# ==============================

def build_pairs(emb_a, emb_b, labels, ages, indices):
    pairs  = []
    truths = []

    idx = list(indices)

    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            ii = idx[i]
            jj = idx[j]

            same_person = (labels[ii] == labels[jj])
            age_gap     = abs(int(ages[ii]) - int(ages[jj]))

            if same_person and age_gap < MIN_AGE_GAP:
                continue

            # Concatenate both model embeddings as input
            feat = np.concatenate([emb_a[ii], emb_b[jj]])
            pairs.append(feat)
            truths.append(1 if same_person else 0)

    return np.array(pairs, dtype=np.float32), np.array(truths, dtype=np.float32)

# ==============================
# HYBRID FUSION NETWORK
# Input: concat of both embeddings
# Output: similarity score [0,1]
# ==============================

class HybridFusionNet(nn.Module):
    def __init__(self, dim_a=512, dim_b=512):
        super().__init__()
        input_dim = dim_a + dim_b   # 1024

        # Use LayerNorm instead of BatchNorm1d
        # LayerNorm works with any batch size including 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ==============================
# TRAIN HYBRID MODEL
# ==============================

def train_hybrid(X_train, y_train, X_test, y_test):
    print(f"\n  Train pairs : {len(X_train)}")
    print(f"  Test  pairs : {len(X_test)}")

    # Convert to tensors
    X_tr = torch.tensor(X_train).to(DEVICE)
    y_tr = torch.tensor(y_train).to(DEVICE)
    X_te = torch.tensor(X_test).to(DEVICE)
    y_te = torch.tensor(y_test).to(DEVICE)

    # Dataset & DataLoader
    train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True  # avoid batch size 1 error
    )

    # Model
    input_dim = X_train.shape[1]
    dim_each  = input_dim // 2
    model     = HybridFusionNet(dim_a=dim_each, dim_b=dim_each).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # Handle class imbalance (many more impostors than genuine)
    pos_count  = int(y_train.sum())
    neg_count  = len(y_train) - pos_count
    ratio      = neg_count / max(pos_count, 1)
    criterion  = nn.BCELoss()
    print(f"  Class ratio (impostor/genuine): {ratio:.1f}x")

    train_losses = []
    test_losses  = []

    print(f"\n  Training Hybrid Model — {EPOCHS} epochs")
    print(f"  {'Epoch':<8} {'Train Loss':<15} {'Test Loss':<15} {'Test Acc'}")
    print(f"  {'-'*50}")

    best_loss  = float('inf')
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):

        # --- Train ---
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_dl)
        train_losses.append(train_loss)

        # --- Evaluate ---
        model.eval()
        with torch.no_grad():
            te_preds = model(X_te)
            te_loss  = criterion(te_preds, y_te).item()
            te_acc   = ((te_preds > 0.5).float() == y_te).float().mean().item()

        test_losses.append(te_loss)
        scheduler.step()

        # Save best model
        if te_loss < best_loss:
            best_loss  = te_loss
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(MODEL_SAVE, "hybrid_best.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:<4} | Loss: {train_loss:.4f}        "
                  f"| Test Loss: {te_loss:.4f}     | Acc: {te_acc*100:.2f}%")

    print(f"\n  ✅ Best model saved at epoch {best_epoch} "
          f"(Test Loss={best_loss:.4f})")

    return model, train_losses, test_losses

# ==============================
# COMPUTE METRICS ON TEST SET
# ==============================

def compute_metrics(model, X_test, y_test):
    model.eval()
    X_te = torch.tensor(X_test).to(DEVICE)

    with torch.no_grad():
        scores = model(X_te).cpu().numpy()

    fpr, tpr, _ = roc_curve(y_test, scores)
    fnr         = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer     = float(fpr[eer_idx])

    def get_fmr(target_fnmr):
        target = 1.0 / target_fnmr
        idx    = np.where(fnr <= target)[0]
        return float(fpr[idx[-1]]) if len(idx) > 0 else 1.0

    def get_tar(target_far):
        idx = np.where(fpr <= target_far)[0]
        return float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

    metrics = {
        "EER"          : round(eer, 6),
        "FMR10"        : round(get_fmr(10),    6),
        "FMR100"       : round(get_fmr(100),   6),
        "FMR1000"      : round(get_fmr(1000),  6),
        "FMR10000"     : round(get_fmr(10000), 6),
        "TAR@FAR=1%"   : round(get_tar(0.01),  6),
        "TAR@FAR=0.1%" : round(get_tar(0.001), 6),
    }

    return metrics, fpr, tpr, fnr

# ==============================
# SAVE ALL PLOTS
# ==============================

def save_plots(train_losses, test_losses,
               fpr_a, tpr_a, fpr_b, tpr_b,
               fpr_h, tpr_h,
               eer_a, eer_b, eer_h):

    # --- Training Loss Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses,  label='Test Loss',  linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Hybrid Model Training — Loss Curve (80/20 Split)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, "hybrid_loss_curve.png"), dpi=150)
    plt.close()
    print("  Saved: hybrid_loss_curve.png")

    # --- ROC Comparison ---
    plt.figure(figsize=(10, 7))
    plt.plot(fpr_a, tpr_a,
             label=f"buffalo_l   (EER={eer_a:.4f})",
             linewidth=2, linestyle='--')
    plt.plot(fpr_b, tpr_b,
             label=f"buffalo_m   (EER={eer_b:.4f})",
             linewidth=2, linestyle='--')
    plt.plot(fpr_h, tpr_h,
             label=f"Hybrid Model (EER={eer_h:.4f})",
             linewidth=2.5, color='red')
    plt.plot([0,1],[0,1],'k--', alpha=0.3, label="Random")
    plt.xlabel("False Match Rate (FMR)", fontsize=12)
    plt.ylabel("True Match Rate (TMR)",  fontsize=12)
    plt.title("ROC Curve — Hybrid vs Individual Models", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, "hybrid_roc_curve.png"), dpi=150)
    plt.close()
    print("  Saved: hybrid_roc_curve.png")

    # --- EER Bar Chart ---
    models = ['buffalo_l', 'buffalo_m', 'Hybrid']
    eers   = [eer_a, eer_b, eer_h]
    colors = ['steelblue', 'steelblue',
              'green' if eer_h == min(eers) else 'red']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, eers, color=colors, edgecolor='black')
    for bar, val in zip(bars, eers):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.001,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=11)
    plt.ylabel("EER (Lower is Better)", fontsize=11)
    plt.title("EER Comparison — Hybrid vs Individual Models", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, "hybrid_eer_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: hybrid_eer_comparison.png")

# ==============================
# MAIN
# ==============================

print("\n" + "="*60)
print("   HYBRID MODEL TRAINING — buffalo_l + buffalo_m")
print("="*60)
print(f"  Split  : {int(TRAIN_RATIO*100)}% Train / {int((1-TRAIN_RATIO)*100)}% Test")
print(f"  Epochs : {EPOCHS}")
print(f"  LR     : {LR}")
print("="*60)

# ==============================
# STEP 1 — Load Both Models
# ==============================

print("\n[Step 1] Loading InsightFace Models...")
app_a = load_insightface(MODEL_A)
app_b = load_insightface(MODEL_B)

if app_a is None or app_b is None:
    print("❌ Could not load both models. Exiting.")
    exit()

# ==============================
# STEP 2 — Extract Embeddings
# ==============================

print(f"\n[Step 2] Extracting Embeddings...")
emb_a, labels_a, ages_a = extract_embeddings(app_a, MODEL_A)
emb_b, labels_b, ages_b = extract_embeddings(app_b, MODEL_B)

# Align to common images
# Use labels_a as reference (both models see same images)
# Find common indices
common_mask = np.isin(labels_a, labels_b)
emb_a  = emb_a[common_mask]
labels = labels_a[common_mask]
ages   = ages_a[common_mask]

# Re-extract emb_b aligned to same order
emb_b_aligned = []
for i, (pid, age) in enumerate(zip(labels, ages)):
    match = np.where((labels_b == pid) & (ages_b == age))[0]
    if len(match) > 0:
        emb_b_aligned.append(emb_b[match[0]])
    else:
        emb_b_aligned.append(np.zeros(emb_b.shape[1]))

emb_b_aligned = np.array(emb_b_aligned)

print(f"\n  Aligned embeddings: {len(labels)}")

# ==============================
# STEP 3 — 80/20 Split by Person
# Ensures same person not in
# both train and test
# ==============================

print(f"\n[Step 3] Splitting Dataset — 80% Train / 20% Test by Person ID...")

unique_persons = np.unique(labels)
np.random.seed(42)
np.random.shuffle(unique_persons)

split_idx     = int(len(unique_persons) * TRAIN_RATIO)
train_persons = set(unique_persons[:split_idx])
test_persons  = set(unique_persons[split_idx:])

train_indices = np.where(np.isin(labels, list(train_persons)))[0]
test_indices  = np.where(np.isin(labels, list(test_persons)))[0]

print(f"  Total persons  : {len(unique_persons)}")
print(f"  Train persons  : {len(train_persons)} ({len(train_indices)} images)")
print(f"  Test  persons  : {len(test_persons)}  ({len(test_indices)} images)")

# ==============================
# STEP 4 — Build Pair Datasets
# ==============================

print(f"\n[Step 4] Building Pair Datasets...")
print("  Building train pairs...")
X_train, y_train = build_pairs(emb_a, emb_b_aligned, labels, ages, train_indices)

print("  Building test pairs...")
X_test,  y_test  = build_pairs(emb_a, emb_b_aligned, labels, ages, test_indices)

print(f"\n  Train pairs : {len(X_train)} "
      f"(Genuine: {int(y_train.sum())}, "
      f"Impostor: {int((y_train==0).sum())})")
print(f"  Test  pairs : {len(X_test)}  "
      f"(Genuine: {int(y_test.sum())}, "
      f"Impostor: {int((y_test==0).sum())})")

# ==============================
# STEP 5 — Compute Individual
# Model Metrics (Baseline)
# ==============================

print(f"\n[Step 5] Computing Individual Model Baselines...")

def cosine_scores(emb, labels, ages, indices):
    scores = []
    truths = []
    idx    = list(indices)
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            ii, jj    = idx[i], idx[j]
            same      = (labels[ii] == labels[jj])
            age_gap   = abs(int(ages[ii]) - int(ages[jj]))
            if same and age_gap < MIN_AGE_GAP:
                continue
            scores.append(float(np.dot(emb[ii], emb[jj])))
            truths.append(1 if same else 0)
    return np.array(scores), np.array(truths)

print("  Computing buffalo_l test scores...")
scores_a, truths_a = cosine_scores(emb_a, labels, ages, test_indices)
fpr_a, tpr_a, _    = roc_curve(truths_a, scores_a)
fnr_a              = 1 - tpr_a
eer_a              = float(fpr_a[np.nanargmin(np.abs(fnr_a - fpr_a))])

print("  Computing buffalo_m test scores...")
scores_b, truths_b = cosine_scores(emb_b_aligned, labels, ages, test_indices)
fpr_b, tpr_b, _    = roc_curve(truths_b, scores_b)
fnr_b              = 1 - tpr_b
eer_b              = float(fpr_b[np.nanargmin(np.abs(fnr_b - fpr_b))])

print(f"\n  buffalo_l  EER (test): {eer_a:.6f}")
print(f"  buffalo_m  EER (test): {eer_b:.6f}")

# ==============================
# STEP 6 — Train Hybrid Model
# ==============================

print(f"\n[Step 6] Training Hybrid Fusion Model...")
model, train_losses, test_losses = train_hybrid(
    X_train, y_train, X_test, y_test
)

# ==============================
# STEP 7 — Evaluate Hybrid
# ==============================

print(f"\n[Step 7] Evaluating Hybrid Model on Test Set...")

# Load best saved model
best_model = HybridFusionNet(
    dim_a=emb_a.shape[1],
    dim_b=emb_b_aligned.shape[1]
).to(DEVICE)
best_model.load_state_dict(
    torch.load(os.path.join(MODEL_SAVE, "hybrid_best.pt"),
               map_location=DEVICE, weights_only=True)
)

h_metrics, fpr_h, tpr_h, fnr_h = compute_metrics(best_model, X_test, y_test)

print(f"\n  {'='*50}")
print(f"  HYBRID MODEL — FINAL METRIC SCORES (Test Set)")
print(f"  {'='*50}")
print(f"  {'Metric':<20} {'buffalo_l':>12} {'buffalo_m':>12} {'Hybrid':>12}")
print(f"  {'-'*58}")

def get_eer(fpr, fnr):
    return round(float(fpr[np.nanargmin(np.abs(fnr - fpr))]), 6)

metrics_a = {
    "EER"          : round(eer_a, 6),
    "TAR@FAR=1%"   : round(float(tpr_a[np.where(fpr_a <= 0.01)[0][-1]
                            if len(np.where(fpr_a <= 0.01)[0]) > 0
                            else 0]), 6),
    "TAR@FAR=0.1%" : round(float(tpr_a[np.where(fpr_a <= 0.001)[0][-1]
                            if len(np.where(fpr_a <= 0.001)[0]) > 0
                            else 0]), 6),
}
metrics_b = {
    "EER"          : round(eer_b, 6),
    "TAR@FAR=1%"   : round(float(tpr_b[np.where(fpr_b <= 0.01)[0][-1]
                            if len(np.where(fpr_b <= 0.01)[0]) > 0
                            else 0]), 6),
    "TAR@FAR=0.1%" : round(float(tpr_b[np.where(fpr_b <= 0.001)[0][-1]
                            if len(np.where(fpr_b <= 0.001)[0]) > 0
                            else 0]), 6),
}

all_metrics = ["EER", "FMR10", "FMR100", "FMR1000", "FMR10000",
               "TAR@FAR=1%", "TAR@FAR=0.1%"]

for m in all_metrics:
    va = metrics_a.get(m, "N/A")
    vb = metrics_b.get(m, "N/A")
    vh = h_metrics.get(m, "N/A")
    print(f"  {m:<20} {str(va):>12} {str(vb):>12} {str(vh):>12}")

print(f"  {'='*50}")

# ==============================
# STEP 8 — Save Results & Plots
# ==============================

print(f"\n[Step 8] Saving Results...")

# Save metrics CSV
df = pd.DataFrame([
    {"Model": "buffalo_l",   **metrics_a, "Type": "Individual"},
    {"Model": "buffalo_m",   **metrics_b, "Type": "Individual"},
    {"Model": "Hybrid",      **h_metrics, "Type": "Hybrid"},
])
csv_path = os.path.join(RESULT_PATH, "hybrid_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"  Saved: hybrid_metrics.csv")

# Save plots
save_plots(
    train_losses, test_losses,
    fpr_a, tpr_a,
    fpr_b, tpr_b,
    fpr_h, tpr_h,
    eer_a, eer_b, h_metrics["EER"]
)

# Save recommendation
rec_path = os.path.join(RESULT_PATH, "hybrid_recommendation.txt")
with open(rec_path, "w", encoding="utf-8") as f:
    f.write("HYBRID MODEL TRAINING RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Split         : 80% Train / 20% Test\n")
    f.write(f"Epochs        : {EPOCHS}\n")
    f.write(f"Models Fused  : {MODEL_A} + {MODEL_B}\n\n")
    f.write(f"buffalo_l EER : {eer_a:.6f}\n")
    f.write(f"buffalo_m EER : {eer_b:.6f}\n")
    f.write(f"Hybrid    EER : {h_metrics['EER']:.6f}\n\n")

    if h_metrics['EER'] < min(eer_a, eer_b):
        f.write("[OK] Hybrid model OUTPERFORMS both individual models!\n")
        f.write("-> Use Hybrid for UIDAI submission.\n")
    else:
        best = MODEL_A if eer_a < eer_b else MODEL_B
        f.write(f"[INFO] Individual model '{best}' performs better.\n")
        f.write(f"-> Use '{best}' for UIDAI submission.\n")

print(f"  Saved: hybrid_recommendation.txt")
print(f"\n{'='*60}")
print(f"  Hybrid Training Complete!")
print(f"  Model saved : {MODEL_SAVE}\\hybrid_best.pt")
print(f"  Results     : {RESULT_PATH}")

# Final verdict
print(f"\n{'='*60}")
print(f"  FINAL VERDICT")
print(f"{'='*60}")
print(f"  buffalo_l  EER : {eer_a:.6f}")
print(f"  buffalo_m  EER : {eer_b:.6f}")
print(f"  Hybrid     EER : {h_metrics['EER']:.6f}")

if h_metrics['EER'] < min(eer_a, eer_b):
    improvement = round((min(eer_a, eer_b) - h_metrics['EER']) / min(eer_a, eer_b) * 100, 2)
    print(f"\n  [BETTER] Hybrid improved by {improvement}%")
    print(f"  -> Use Hybrid Model for UIDAI submission")
else:
    print(f"\n  [INFO] Individual model is still better")
    print(f"  -> Use buffalo_l for UIDAI submission")
print(f"{'='*60}")