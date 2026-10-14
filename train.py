"""
UIDAI Face Matching — GPU Training Script
Dataset: FGNET (age variation 5+ years)
Pipeline: Real-ESRGAN 2x + CodeFormer -> SCRFD + IResNet50/ArcFace
Results saved to: FGNET/result/

FIXES APPLIED:
  1. IResNet50 img_size passed → fixes shape mismatch (100352 vs 32768)
  2. workers=0 → fixes repeated startup logs on Windows
  3. GPU auto-detection with clear fix instructions
  4. Startup log moved inside __main__ → only prints once
"""

import os
import sys
import json
import time
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from dataset import FGNETDataset
from models import IResNet50, ArcFaceLoss, cosine_similarity, export_to_onnx, quantize_to_int8
from enhancement import EnhancementPipeline

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
_candidate_fgnet = os.path.join(_PARENT_DIR, "FGNET (1)", "FGNET")
if os.path.exists(_candidate_fgnet):
    BASE_DIR = _candidate_fgnet
else:
    BASE_DIR = os.path.join(_SCRIPT_DIR, "FGNET")

IMAGES_DIR = os.path.join(BASE_DIR, "images")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
log_file = os.path.join(RESULT_DIR, "training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CONFIG  (device/fp16/workers set in __main__ after GPU check)
# ─────────────────────────────────────────
CONFIG = {
    "images_dir":        IMAGES_DIR,
    "result_dir":        RESULT_DIR,
    "embedding_dim":     512,
    "img_size":          112,
    "epochs":            60,
    "batch_size":        32,
    "lr":                1e-3,
    "weight_decay":      5e-4,
    "val_split":         0.15,
    "arcface_s":         64.0,
    "arcface_m":         0.5,
    "quality_threshold": 100.0,
    "esrgan_scale":      2,
    "codeformer_w":      1.0,
    "match_threshold":   0.5,
    # These three are overridden at runtime:
    "device":            "cpu",
    "fp16":              False,
    "workers":           0,     # MUST stay 0 on Windows
}


# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    return float(fpr[idx]), float(thresholds[idx])


def compute_fmr_at_rate(scores, labels, fmr_rate):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.searchsorted(fpr, 1.0 / fmr_rate)
    return float(fnr[min(idx, len(fnr) - 1)])


def evaluate_verification(model, dataset, device):
    model.eval()
    pairs  = dataset.get_pair_samples()
    scores, labels = [], []

    with torch.no_grad():
        for path1, path2, label in pairs:
            try:
                t1 = dataset.load_image_tensor(path1).to(device)
                t2 = dataset.load_image_tensor(path2).to(device)
                e1 = model(t1).cpu().numpy()[0]
                e2 = model(t2).cpu().numpy()[0]
                scores.append(cosine_similarity(e1, e2))
                labels.append(label)
            except Exception:
                continue

    scores = np.array(scores)
    labels = np.array(labels)

    eer, eer_thresh = compute_eer(scores, labels)
    fpr, tpr, _     = roc_curve(labels, scores)
    roc_auc         = auc(fpr, tpr)

    metrics = {
        "EER":           round(eer, 6),
        "EER_threshold": round(eer_thresh, 6),
        "FMR10000_FNMR": round(compute_fmr_at_rate(scores, labels, 10000), 6),
        "FMR1000_FNMR":  round(compute_fmr_at_rate(scores, labels, 1000),  6),
        "FMR100_FNMR":   round(compute_fmr_at_rate(scores, labels, 100),   6),
        "FMR10_FNMR":    round(compute_fmr_at_rate(scores, labels, 10),    6),
        "ROC_AUC":       round(float(roc_auc), 6),
        "num_pairs":     len(pairs),
    }
    return metrics, scores, labels, fpr, tpr


def plot_roc_det(fpr, tpr, eer, epoch, result_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC={roc_auc:.4f}")
    ax1.plot([0,1],[0,1],"k--",lw=1)
    ax1.scatter([eer],[1-eer],color="red",zorder=5,label=f"EER={eer:.4f}")
    ax1.set_xlabel("FMR"); ax1.set_ylabel("1-FNMR")
    ax1.set_title(f"ROC — Epoch {epoch}")
    ax1.legend(); ax1.grid(True)

    fnr = 1 - tpr
    ax2.plot(fpr, fnr, color="steelblue", lw=2)
    ax2.set_xlabel("FMR"); ax2.set_ylabel("FNMR")
    ax2.set_title(f"DET — Epoch {epoch}")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.grid(True, which="both", ls="--")

    plt.tight_layout()
    path = os.path.join(result_dir, f"roc_det_epoch_{epoch}.png")
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"ROC/DET saved -> {path}")


# ─────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────

class UIDAITrainer:

    def __init__(self, config):
        self.cfg        = config
        self.device     = config["device"]
        self.result_dir = config["result_dir"]

        # Dataset
        log.info("Loading FGNET dataset...")
        full_dataset = FGNETDataset(
            images_dir=config["images_dir"],
            img_size=config["img_size"],
            augment=True, min_age_gap=5, mode="train"
        )
        self.num_classes = full_dataset.get_num_classes()
        log.info(f"Identities: {self.num_classes} | Images: {len(full_dataset)}")

        val_size   = int(len(full_dataset) * config["val_split"])
        train_size = len(full_dataset) - val_size
        self.train_ds, self.val_ds = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        self.val_ds.dataset.mode = "val"

        self.train_loader = DataLoader(
            self.train_ds, batch_size=config["batch_size"],
            shuffle=True, num_workers=config["workers"],
            pin_memory=(self.device=="cuda"), drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=config["batch_size"],
            shuffle=False, num_workers=config["workers"],
            pin_memory=(self.device=="cuda")
        )
        log.info(f"Train: {train_size} | Val: {val_size}")

        # ── FIX: pass img_size → computes correct FC flatten dim ──
        self.model = IResNet50(
            embedding_dim=config["embedding_dim"],
            dropout=0.4,
            img_size=config["img_size"]    # <-- THIS WAS THE BUG
        ).to(self.device)

        self.arcface = ArcFaceLoss(
            embedding_dim=config["embedding_dim"],
            num_classes=self.num_classes,
            s=config["arcface_s"], m=config["arcface_m"]
        ).to(self.device)

        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.arcface.parameters()),
            lr=config["lr"], weight_decay=config["weight_decay"]
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"], eta_min=1e-6
        )
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if config["fp16"] and self.device == "cuda" else None
        )

        log.info("Loading enhancement pipeline...")
        self.enhancer = EnhancementPipeline(
            device=self.device,
            quality_threshold=config["quality_threshold"]
        )

        self.history = {
            "train_loss":[], "val_loss":[],
            "eer":[], "roc_auc":[],
            "fmr10000":[], "fmr1000":[], "fmr100":[], "fmr10":[]
        }
        self.best_eer         = float("inf")
        self.full_dataset_ref = full_dataset

    def train_epoch(self, epoch):
        self.model.train(); self.arcface.train()
        total_loss = 0.0
        n = len(self.train_loader)

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast('cuda'):
                    emb  = self.model(images)
                    loss = self.arcface(emb, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                emb  = self.model(images)
                loss = self.arcface(emb, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            total_loss += loss.item()
            if i % 20 == 0:
                log.info(f"Epoch [{epoch}/{self.cfg['epochs']}] Batch [{i}/{n}] Loss: {loss.item():.4f}")

        return total_loss / n

    def validate_epoch(self):
        self.model.eval(); self.arcface.eval()
        total = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                emb  = self.model(images)
                loss = self.arcface(emb, labels)
                total += loss.item()
        return total / len(self.val_loader)

    def save_checkpoint(self, epoch, metrics, is_best=False):
        name = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        path = os.path.join(self.result_dir, name)
        torch.save({
            "epoch": epoch, "model_state": self.model.state_dict(),
            "arcface_state": self.arcface.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics, "config": self.cfg,
            "num_classes": self.num_classes,
        }, path)
        log.info(f"Checkpoint saved -> {path}")

    def export_models(self):
        onnx = os.path.join(self.result_dir, "face_recognition.onnx")
        int8 = os.path.join(self.result_dir, "face_recognition_int8.onnx")
        export_to_onnx(self.model, onnx, img_size=self.cfg["img_size"], device=self.device)
        quantize_to_int8(onnx, int8)

    def save_metrics(self, metrics, epoch):
        with open(os.path.join(self.result_dir, f"metrics_epoch_{epoch}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(self.result_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0,0].plot(self.history["train_loss"], label="Train")
        axes[0,0].plot(self.history["val_loss"],   label="Val")
        axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(True)
        axes[0,1].plot(self.history["eer"], color="red", label="EER")
        axes[0,1].set_title("EER (lower=better)"); axes[0,1].legend(); axes[0,1].grid(True)
        axes[1,0].plot(self.history["fmr10000"], label="FMR10000")
        axes[1,0].plot(self.history["fmr1000"],  label="FMR1000")
        axes[1,0].set_title("FMR Metrics"); axes[1,0].legend(); axes[1,0].grid(True)
        axes[1,1].plot(self.history["roc_auc"], color="green", label="AUC")
        axes[1,1].set_title("ROC AUC (higher=better)"); axes[1,1].legend(); axes[1,1].grid(True)
        plt.tight_layout()
        path = os.path.join(self.result_dir, "training_curves.png")
        plt.savefig(path, dpi=150); plt.close()
        log.info(f"Training curves -> {path}")

    def train(self):
        log.info(f"Training for {self.cfg['epochs']} epochs on {self.device}...")
        t0 = time.time()

        for epoch in range(1, self.cfg["epochs"] + 1):
            te = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate_epoch()
            self.scheduler.step()

            metrics = {}
            if epoch % 5 == 0 or epoch == self.cfg["epochs"]:
                log.info(f"Evaluating verification at epoch {epoch}...")
                metrics, scores, labels, fpr, tpr = evaluate_verification(
                    self.model, self.full_dataset_ref, self.device
                )
                plot_roc_det(fpr, tpr, metrics["EER"], epoch, self.result_dir)
                self.save_metrics(metrics, epoch)
                log.info(
                    f"EER={metrics['EER']:.4f} | "
                    f"FMR10000={metrics['FMR10000_FNMR']:.4f} | "
                    f"AUC={metrics['ROC_AUC']:.4f}"
                )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            if metrics:
                self.history["eer"].append(metrics["EER"])
                self.history["roc_auc"].append(metrics["ROC_AUC"])
                self.history["fmr10000"].append(metrics["FMR10000_FNMR"])
                self.history["fmr1000"].append(metrics["FMR1000_FNMR"])
                self.history["fmr100"].append(metrics["FMR100_FNMR"])
                self.history["fmr10"].append(metrics["FMR10_FNMR"])

            log.info(
                f"Epoch {epoch}/{self.cfg['epochs']} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Time: {round(time.time()-te,1)}s"
            )

            if metrics and metrics["EER"] < self.best_eer:
                self.best_eer = metrics["EER"]
                self.save_checkpoint(epoch, metrics, is_best=True)
                log.info(f"New best EER: {self.best_eer:.4f}")

            if epoch % 10 == 0:
                self.save_checkpoint(epoch, metrics)

        self.export_models()
        self.plot_training_curves()
        log.info(f"Done in {round((time.time()-t0)/60,1)} min | Best EER: {self.best_eer:.4f}")
        log.info(f"Results: {self.result_dir}")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Windows fix: stops worker subprocesses re-running this block
    import multiprocessing
    multiprocessing.freeze_support()

    # ── GPU CHECK ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SYSTEM CHECK")
    print("="*60)

    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA Available  : {cuda_ok}")

    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"  GPU             : {gpu_name}")
        print(f"  GPU Memory      : {gpu_mem} GB")
        CONFIG["device"]     = "cuda"
        CONFIG["fp16"]       = True
        CONFIG["workers"]    = 0       # Keep 0 on Windows even with CUDA
        CONFIG["batch_size"] = 32
        print("  Mode            : GPU + FP16  (fast)")
    else:
        print("  WARNING: No CUDA GPU detected!")
        print()
        print("  HOW TO FIX:")
        print("  Step 1 - Check driver     : nvidia-smi")
        print("  Step 2 - Reinstall torch  :")
        print("           pip uninstall torch torchvision -y")
        print("           pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("  Step 3 - Verify           :")
        print("           python -c \"import torch; print(torch.cuda.is_available())\"")
        print()
        print("  Running on CPU for now (training will be slow)...")
        CONFIG["device"]     = "cpu"
        CONFIG["fp16"]       = False
        CONFIG["workers"]    = 0
        CONFIG["batch_size"] = 16   # smaller batch on CPU

    print("="*60 + "\n")

    # ── START ────────────────────────────────────────────────────
    log.info("="*60)
    log.info("UIDAI Face Matching — Training Started")
    log.info(f"Device  : {CONFIG['device']}")
    log.info(f"Dataset : {IMAGES_DIR}")
    log.info(f"Results : {RESULT_DIR}")
    log.info("="*60)

    trainer = UIDAITrainer(CONFIG)
    trainer.train()