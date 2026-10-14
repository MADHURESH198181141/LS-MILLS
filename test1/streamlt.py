"""
UIDAI Face Biometric Verification System
==========================================
Pipeline (mirrors training exactly):
  Image → InsightFace buffalo_l  → 512-d emb_a  ─┐
  Image → InsightFace buffalo_m  → 512-d emb_b  ─┴→ concat 1024-d → HybridFusionNet → score [0,1]

Model  : hybrid_best.pt   (HybridFusionNet state-dict)
Embedders: InsightFace buffalo_l + buffalo_m
"""

import os, ssl, warnings, sys
import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
from PIL import Image

# ── suppress SSL / insightface warnings ─────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["CURL_CA_BUNDLE"]     = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import requests, urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig = requests.Session.request
    def _patched(self, *a, **kw):
        kw["verify"] = False
        return _orig(self, *a, **kw)
    requests.Session.request = _patched
except Exception:
    pass

# ── InsightFace ──────────────────────────────────────────────────────────────
try:
    from insightface.app import FaceAnalysis
    from sklearn.preprocessing import normalize
    INSIGHTFACE_OK = True
except ImportError:
    INSIGHTFACE_OK = False

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="UIDAI Face Biometric",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Rajdhani',sans-serif;}
.stApp{background:#0d1117;color:#e6edf3;}
.uidai-header{
    background:linear-gradient(135deg,#1a237e 0%,#0288d1 60%,#00838f 100%);
    border-radius:12px;padding:26px 34px;margin-bottom:22px;
}
.uidai-header h1{margin:0;font-size:2rem;font-weight:700;color:#fff;letter-spacing:2px;}
.uidai-header p{margin:4px 0 0;font-size:.9rem;color:rgba(255,255,255,.72);
                font-family:'IBM Plex Mono',monospace;}
.metric-card{background:#161b22;border:1px solid #30363d;border-radius:10px;
             padding:18px;text-align:center;}
.metric-value{font-size:2rem;font-weight:700;font-family:'IBM Plex Mono',monospace;}
.metric-label{font-size:.82rem;color:#8b949e;margin-top:4px;}
.result-match{background:linear-gradient(90deg,#1b5e20,#2e7d32);
              border-left:6px solid #66bb6a;border-radius:8px;
              padding:18px 26px;font-size:1.35rem;font-weight:700;
              color:#e8f5e9;letter-spacing:1px;}
.result-no-match{background:linear-gradient(90deg,#7f0000,#c62828);
                 border-left:6px solid #ef9a9a;border-radius:8px;
                 padding:18px 26px;font-size:1.35rem;font-weight:700;
                 color:#ffebee;letter-spacing:1px;}
.section-title{font-size:.95rem;font-weight:700;color:#58a6ff;letter-spacing:3px;
               text-transform:uppercase;border-bottom:1px solid #30363d;
               padding-bottom:7px;margin:22px 0 14px;}
.info-box{background:#0d2137;border:1px solid #0288d1;border-radius:8px;
          padding:11px 15px;font-size:.8rem;color:#90caf9;
          font-family:'IBM Plex Mono',monospace;line-height:1.9;}
.warn-box{background:#1c1200;border:1px solid #ff8f00;border-radius:8px;
          padding:11px 15px;font-size:.8rem;color:#ffe082;
          font-family:'IBM Plex Mono',monospace;line-height:1.7;}
.err-box{background:#1c0000;border:1px solid #ef5350;border-radius:8px;
         padding:11px 15px;font-size:.82rem;color:#ffcdd2;
         font-family:'IBM Plex Mono',monospace;}
.step-badge{display:inline-block;background:#0288d1;color:#fff;border-radius:50%;
            width:24px;height:24px;text-align:center;line-height:24px;
            font-weight:700;margin-right:7px;font-size:.82rem;}
[data-testid="stSidebar"]{background:#161b22 !important;border-right:1px solid #30363d;}
.stProgress>div>div{background:linear-gradient(90deg,#0288d1,#00838f);}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PATHS  (exact paths from your training script)
# ════════════════════════════════════════════════════════════════════════════
_PARENT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE     = os.path.join(_PARENT_DIR, "hybrid_model")
MODEL_PT_PATH  = os.path.join(MODEL_SAVE, "hybrid_best.pt")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ════════════════════════════════════════════════════════════════════════════
# EXACT HybridFusionNet — copy-pasted from your training code
# ════════════════════════════════════════════════════════════════════════════
class HybridFusionNet(nn.Module):
    """
    Exact replica of the architecture used during training.
    Input  : concat of buffalo_l (512-d) + buffalo_m (512-d)  =  1024-d
    Output : scalar similarity score in [0, 1]
    """
    def __init__(self, dim_a=512, dim_b=512):
        super().__init__()
        input_dim = dim_a + dim_b   # 1024

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


# ════════════════════════════════════════════════════════════════════════════
# LOAD INSIGHTFACE MODELS
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_insightface_models():
    """
    Load buffalo_l + buffalo_m with:
      - det_size=(320,320)  → faster than default 640x640
      - det_thresh=0.30     → more lenient face detection
    """
    if not INSIGHTFACE_OK:
        return None, None, "InsightFace not installed"

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if DEVICE == "cuda" else ["CPUExecutionProvider"]
    )
    errors = []
    app_l, app_m = None, None
    try:
        app_l = FaceAnalysis(name="buffalo_l", providers=providers)
        # 320×320 det is 4× faster than 640×640, still accurate enough
        app_l.prepare(ctx_id=0 if DEVICE == "cuda" else -1,
                      det_size=(320, 320), det_thresh=0.30)
    except Exception as e:
        errors.append(f"buffalo_l: {e}")

    try:
        app_m = FaceAnalysis(name="buffalo_m", providers=providers)
        app_m.prepare(ctx_id=0 if DEVICE == "cuda" else -1,
                      det_size=(320, 320), det_thresh=0.30)
    except Exception as e:
        errors.append(f"buffalo_m: {e}")

    if errors:
        return app_l, app_m, " | ".join(errors)
    return app_l, app_m, "OK"


# ════════════════════════════════════════════════════════════════════════════
# LOAD HybridFusionNet
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_hybrid_model(pt_path: str):
    """Load HybridFusionNet weights from hybrid_best.pt."""
    if not os.path.exists(pt_path):
        return None, f"File not found: {pt_path}"
    try:
        model = HybridFusionNet(dim_a=512, dim_b=512)
        state = torch.load(pt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval().to(DEVICE)
        return model, "OK"
    except Exception as e:
        return None, str(e)


# ════════════════════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION  (mirrors training pipeline exactly)
# ════════════════════════════════════════════════════════════════════════════
def get_embedding(app, img_bgr: np.ndarray):
    """
    Extract normalised 512-d embedding from an InsightFace model.
    Multi-scale retry logic:
      1. Try original image
      2. If no face → upscale to 640px (helps small/low-res images)
      3. If still no face → try grayscale→RGB conversion
    Returns (embedding, face_bbox) or (None, None) if no face found.
    """
    def _best_face(faces):
        if not faces:
            return None
        return sorted(faces,
                      key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                      reverse=True)[0]

    def _to_emb(face, original_img, scale=1.0):
        emb  = normalize(face.embedding.reshape(1, -1))[0].astype(np.float32)
        bbox = (face.bbox / scale).astype(int)   # rescale bbox back
        return emb, bbox

    # ── Attempt 1: original size ─────────────────────────────────────
    faces = app.get(img_bgr)
    face  = _best_face(faces)
    if face is not None:
        return _to_emb(face, img_bgr, scale=1.0)

    # ── Attempt 2: upscale short-side to 640 ─────────────────────────
    h, w  = img_bgr.shape[:2]
    short = min(h, w)
    if short < 640:
        scale  = 640.0 / short
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        img_up = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        faces  = app.get(img_up)
        face   = _best_face(faces)
        if face is not None:
            return _to_emb(face, img_up, scale=scale)

    # ── Attempt 3: CLAHE contrast enhancement ────────────────────────
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l     = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
    faces = app.get(enhanced)
    face  = _best_face(faces)
    if face is not None:
        return _to_emb(face, enhanced, scale=1.0)

    return None, None


def run_hybrid_inference(model, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Concatenate buffalo_l + buffalo_m embeddings and pass through HybridFusionNet.
    Returns similarity score in [0, 1].
    """
    feat = np.concatenate([emb_a, emb_b]).astype(np.float32)   # 1024-d
    t    = torch.tensor(feat).unsqueeze(0).to(DEVICE)           # [1, 1024]
    with torch.no_grad():
        score = model(t).item()
    return float(score)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def draw_bbox_on_image(img_rgb: np.ndarray, bbox, color=(0, 220, 100), thickness=3):
    out = img_rgb.copy()
    if bbox is not None:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def segment_cosine(f1, f2, n=10):
    """Per-segment cosine similarity for bar chart."""
    size = min(len(f1), len(f2))
    step = size // n
    sims = []
    for i in range(n):
        a = f1[i*step:(i+1)*step]
        b = f2[i*step:(i+1)*step]
        s = float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))
        sims.append(s)
    return sims


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
for k, v in [("hybrid_model", None), ("app_l", None), ("app_m", None),
             ("models_ready", False), ("verify_result", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="uidai-header">
  <div>
    <h1>🪪 UIDAI FACE BIOMETRIC SYSTEM</h1>
    <p>buffalo_l ⊕ buffalo_m → HybridFusionNet &nbsp;|&nbsp; hybrid_best.pt</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔧 Navigation")
    page = st.radio("Module", ["🏠 Dashboard", "🚀 Load Models", "🔍 Verify Identity"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    THRESHOLD = 0.45
    st.slider("Match Threshold", 0.30, 0.99, THRESHOLD, 0.01, disabled=True,
              help="HybridFusionNet outputs [0,1]. Score ≥ threshold → MATCH (Fixed optimal threshold: 0.45)")
    st.caption("🔒 Threshold locked to optimal recommendation (0.45)")
    st.markdown("---")

    ready = st.session_state.models_ready
    r     = st.session_state.verify_result
    score_disp = f"{r['score']:.4f}" if r else "—"
    col_disp   = "#66bb6a" if (r and r["match"]) else "#ef5350" if r else "#8b949e"

    st.markdown(f"""<div class="info-box">
    InsightFace : {'✅' if INSIGHTFACE_OK else '❌ not installed'}<br>
    buffalo_l   : {'✅' if st.session_state.app_l else '❌'}<br>
    buffalo_m   : {'✅' if st.session_state.app_m else '❌'}<br>
    HybridNet   : {'✅' if st.session_state.hybrid_model else '❌'}<br>
    Device      : {DEVICE.upper()}<br>
    Threshold   : {THRESHOLD}<br>
    Last Score  : <span style="color:{col_disp}">{score_disp}</span>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown('<div class="section-title">System Overview</div>', unsafe_allow_html=True)

    r  = st.session_state.verify_result
    ok = st.session_state.models_ready
    gc = "#66bb6a" if (r and r["match"]) else "#ef5350" if r else "#8b949e"

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{'#66bb6a' if ok else '#ef5350'}">{'✅' if ok else '❌'}</div>
            <div class="metric-label">Models Ready</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1rem;color:#f0883e">buffalo_l<br>+ buffalo_m</div>
            <div class="metric-label">Embedders</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{gc}">{f"{r['score']:.4f}" if r else '—'}</div>
            <div class="metric-label">Last Score</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1.1rem;color:{gc}">
            {("✅ MATCH" if r["match"] else "❌ MISMATCH") if r else "—"}</div>
            <div class="metric-label">Last Verdict</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Exact Inference Pipeline</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info-box">
    📁 Model file  : {MODEL_PT_PATH}<br>
    🧠 Architecture: HybridFusionNet (1024→256→128→64→1)<br>
    🔧 Embedder A  : InsightFace buffalo_l  → 512-d  L2-normalised<br>
    🔧 Embedder B  : InsightFace buffalo_m  → 512-d  L2-normalised<br>
    🔗 Fusion      : concat([emb_l, emb_m]) = 1024-d → HybridFusionNet<br>
    📊 Output      : Sigmoid score  [0.0 → 1.0]<br>
    ✅ Match if    : score ≥ {THRESHOLD}
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    for i,(t,d) in enumerate([
        ("Load Models",    "Loads hybrid_best.pt weights + both InsightFace embedders (buffalo_l, buffalo_m)."),
        ("Upload Images",  "Upload OLD (Aadhaar enrolled) and NEW (live/current) face photos."),
        ("Extract Embeddings", "Each image → buffalo_l gives 512-d emb_a, buffalo_m gives 512-d emb_b."),
        ("Fusion",         "concat([emb_a, emb_b]) = 1024-d vector fed to HybridFusionNet."),
        ("Score",          "HybridFusionNet outputs a Sigmoid score [0,1]. Score ≥ threshold → MATCH."),
        ("Visualise",      "Bar chart shows per-segment similarity + score gauge + embedding overlay."),
    ], 1):
        st.markdown(f'<p><span class="step-badge">{i}</span><strong>{t}</strong> — {d}</p>',
                    unsafe_allow_html=True)

    if not INSIGHTFACE_OK:
        st.markdown("""<div class="err-box">
        ❌ InsightFace not installed.<br>
        Run: <b>pip install insightface onnxruntime</b>  (CPU)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>pip install insightface onnxruntime-gpu</b> (GPU)
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: LOAD MODELS
# ════════════════════════════════════════════════════════════════════════════
elif "Load" in page:
    st.markdown('<div class="section-title">Load All Models</div>', unsafe_allow_html=True)

    if not INSIGHTFACE_OK:
        st.error("❌ InsightFace not installed.\n\n"
                 "Run:  `pip install insightface onnxruntime`")
        st.stop()

    st.markdown(f"""<div class="info-box">
    Will load 3 components:<br>
    1️⃣  InsightFace <b>buffalo_l</b>  (512-d embedder)<br>
    2️⃣  InsightFace <b>buffalo_m</b>  (512-d embedder)<br>
    3️⃣  <b>HybridFusionNet</b> weights from:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;<code>{MODEL_PT_PATH}</code>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([2,1])
    with col_l:
        pt_path = st.text_input("hybrid_best.pt path", value=MODEL_PT_PATH)
    with col_r:
        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Or upload hybrid_best.pt", type=["pt","pth"])

    st.markdown("---")
    if st.button("🚀 Load All Models", type="primary", use_container_width=True):
        prog = st.progress(0)
        log  = st.empty()

        def show(msg):
            log.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)

        # ── Step 1: InsightFace ──────────────────────────────────────────
        show("⏳ Loading InsightFace buffalo_l + buffalo_m ...")
        prog.progress(10)
        app_l, app_m, if_status = load_insightface_models()

        if app_l is None:
            st.error(f"❌ buffalo_l failed to load: {if_status}")
            st.stop()
        if app_m is None:
            st.error(f"❌ buffalo_m failed to load: {if_status}")
            st.stop()

        st.session_state.app_l = app_l
        st.session_state.app_m = app_m
        prog.progress(55)
        show("✅ buffalo_l loaded<br>✅ buffalo_m loaded<br>⏳ Loading HybridFusionNet weights...")

        # ── Step 2: HybridFusionNet ──────────────────────────────────────
        if uploaded:
            tmp = Path("/tmp") / uploaded.name
            tmp.write_bytes(uploaded.read())
            resolve_path = str(tmp)
        else:
            resolve_path = pt_path

        hybrid_model, hm_status = load_hybrid_model(resolve_path)
        prog.progress(90)

        if hybrid_model is None:
            st.error(f"❌ HybridFusionNet failed: {hm_status}")
            st.stop()

        st.session_state.hybrid_model  = hybrid_model
        st.session_state.models_ready  = True
        prog.progress(100)

        # ── Probe: dummy forward pass ─────────────────────────────────
        dummy  = torch.zeros(1, 1024).to(DEVICE)
        with torch.no_grad():
            out = hybrid_model(dummy)
        show(
            "✅ buffalo_l loaded (512-d)<br>"
            "✅ buffalo_m loaded (512-d)<br>"
            "✅ HybridFusionNet loaded (1024→256→128→64→1)<br>"
            f"✅ Dummy forward pass OK — output shape: {tuple(out.shape)}"
        )
        st.success("🎉 All models ready! Go to  **🔍 Verify Identity**")

    # Show current status
    if st.session_state.models_ready:
        st.markdown('<div class="section-title">Current Status</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="info-box">
        ✅ buffalo_l   : loaded<br>
        ✅ buffalo_m   : loaded<br>
        ✅ HybridFusionNet : loaded from {pt_path}<br>
        🖥️  Device     : {DEVICE.upper()}
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: VERIFY IDENTITY
# ════════════════════════════════════════════════════════════════════════════
elif "Verify" in page:
    st.markdown('<div class="section-title">Identity Verification</div>', unsafe_allow_html=True)

    if not st.session_state.models_ready:
        st.warning("⚠️ Models not loaded yet. Go to **🚀 Load Models** first.")
        st.stop()

    app_l        = st.session_state.app_l
    app_m        = st.session_state.app_m
    hybrid_model = st.session_state.hybrid_model

    # ── Upload ────────────────────────────────────────────────────────────
    st.markdown("#### Upload Face Images")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 🗂️ OLD Image (Aadhaar Enrolled)")
        old_file = st.file_uploader("Enrolled face photo",
                                    type=["jpg","jpeg","png"], key="old")
        if old_file:
            old_pil = Image.open(old_file).convert("RGB")
            st.image(old_pil, caption="Enrolled Photo", use_container_width=True)

    with c2:
        st.markdown("##### 📸 NEW Image (Live / Current)")
        new_file = st.file_uploader("Live face photo",
                                    type=["jpg","jpeg","png"], key="new")
        if new_file:
            new_pil = Image.open(new_file).convert("RGB")
            st.image(new_pil, caption="Live Photo", use_container_width=True)

    st.markdown("---")

    if st.button("🔍 Run Biometric Verification", type="primary",
                 use_container_width=True, disabled=not(old_file and new_file)):

        prog_v  = st.progress(0)
        status  = st.empty()

        def vstep(msg, pct):
            status.markdown(f'<div class="info-box">⏳ {msg}</div>',
                            unsafe_allow_html=True)
            prog_v.progress(pct)

        old_bgr = pil_to_bgr(old_pil)
        new_bgr = pil_to_bgr(new_pil)

        # ── buffalo_l embeddings ──────────────────────────────────────
        vstep("buffalo_l → extracting OLD image embedding ...", 10)
        emb_l_old, bbox_l_old = get_embedding(app_l, old_bgr)
        if emb_l_old is None:
            prog_v.empty(); status.empty()
            st.error("❌ buffalo_l: No face detected in OLD image.\n\n"
                     "Tips:\n"
                     "• Use a clear front-facing photo\n"
                     "• Make sure face is well-lit and not blurry\n"
                     "• Avoid heavy occlusions (mask/sunglasses)")
            st.stop()

        vstep("buffalo_l → extracting NEW image embedding ...", 25)
        emb_l_new, bbox_l_new = get_embedding(app_l, new_bgr)
        if emb_l_new is None:
            prog_v.empty(); status.empty()
            st.error("❌ buffalo_l: No face detected in NEW image.\n\n"
                     "Tips:\n"
                     "• Use a clear front-facing photo\n"
                     "• Make sure face is well-lit and not blurry\n"
                     "• Avoid heavy occlusions (mask/sunglasses)")
            st.stop()

        # ── buffalo_m embeddings ──────────────────────────────────────
        vstep("buffalo_m → extracting OLD image embedding ...", 45)
        emb_m_old, bbox_m_old = get_embedding(app_m, old_bgr)
        if emb_m_old is None:
            prog_v.empty(); status.empty()
            st.error("❌ buffalo_m: No face detected in OLD image.")
            st.stop()

        vstep("buffalo_m → extracting NEW image embedding ...", 65)
        emb_m_new, bbox_m_new = get_embedding(app_m, new_bgr)
        if emb_m_new is None:
            prog_v.empty(); status.empty()
            st.error("❌ buffalo_m: No face detected in NEW image.")
            st.stop()

        # ── HybridFusionNet inference ─────────────────────────────────
        vstep("Running HybridFusionNet inference ...", 85)
        score = run_hybrid_inference(hybrid_model, emb_l_old, emb_m_new)
        cos_l = float(np.dot(emb_l_old, emb_l_new))
        cos_m = float(np.dot(emb_m_old, emb_m_new))
        match = score >= THRESHOLD
        st.session_state.verify_result = {"score": score, "match": match}

        prog_v.progress(100)
        status.markdown('<div class="info-box">✅ Inference complete!</div>',
                        unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════
        # RESULTS
        # ════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-title">Verification Result</div>',
                    unsafe_allow_html=True)

        gc = "#66bb6a" if match else "#ef5350"
        if match:
            st.markdown(f"""<div class="result-match">
                ✅ &nbsp; IDENTITY VERIFIED — SAME PERSON<br>
                <span style="font-size:.9rem;font-weight:400">
                HybridFusionNet Score: <b>{score:.4f}</b> &nbsp;|&nbsp;
                Threshold: {THRESHOLD} &nbsp;|&nbsp; Confidence: {score*100:.1f}%
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-no-match">
                ❌ &nbsp; IDENTITY MISMATCH — DIFFERENT PERSON<br>
                <span style="font-size:.9rem;font-weight:400">
                HybridFusionNet Score: <b>{score:.4f}</b> &nbsp;|&nbsp;
                Threshold: {THRESHOLD} &nbsp;|&nbsp; Confidence: {score*100:.1f}%
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Face crops with bounding boxes ──────────────────────────────
        st.markdown('<div class="section-title">Detected Faces</div>',
                    unsafe_allow_html=True)
        ca, cb, cc = st.columns([1,1,1])

        bbox_col = (0,220,80) if match else (220,50,50)
        old_rgb  = np.array(old_pil)
        new_rgb  = np.array(new_pil)

        with ca:
            img_disp = draw_bbox_on_image(old_rgb, bbox_l_old, color=bbox_col)
            fig,ax = plt.subplots(facecolor="#0d1117")
            ax.imshow(img_disp); ax.axis("off"); ax.set_facecolor("#161b22")
            ax.set_title("OLD — Enrolled", color="#e6edf3", fontsize=10, pad=6)
            st.pyplot(fig); plt.close(fig)

        with cb:
            img_disp2 = draw_bbox_on_image(new_rgb, bbox_l_new, color=bbox_col)
            fig,ax = plt.subplots(facecolor="#0d1117")
            ax.imshow(img_disp2); ax.axis("off"); ax.set_facecolor("#161b22")
            ax.set_title("NEW — Live", color="#e6edf3", fontsize=10, pad=6)
            st.pyplot(fig); plt.close(fig)

        with cc:
            # Score gauge
            fig_g, ax_g = plt.subplots(figsize=(3,3), facecolor="#0d1117")
            ax_g.set_facecolor("#161b22")
            theta = np.linspace(0, np.pi, 300)
            ax_g.plot(np.cos(theta), np.sin(theta), color="#30363d", lw=12,
                      solid_capstyle="round")
            fill = np.linspace(0, np.pi*score, 300)
            ax_g.plot(np.cos(fill), np.sin(fill), color=gc, lw=12,
                      solid_capstyle="round")
            th_a = np.pi * THRESHOLD
            ax_g.plot([np.cos(th_a)*.72, np.cos(th_a)*1.12],
                      [np.sin(th_a)*.72, np.sin(th_a)*1.12],
                      color="#ff8f00", lw=2.5)
            ax_g.text(0, .36, f"{score:.3f}", ha="center", va="center",
                      fontsize=22, fontweight="bold", color=gc,
                      fontfamily="monospace")
            ax_g.text(0, .10, "HYBRID SCORE", ha="center",
                      color="#8b949e", fontsize=7)
            ax_g.text(0, -.08, "✅ MATCH" if match else "❌ MISMATCH",
                      ha="center", color=gc, fontsize=10, fontweight="bold")
            ax_g.set_xlim(-1.25,1.25); ax_g.set_ylim(-.25,1.25)
            ax_g.axis("off")
            ax_g.set_title("Score Gauge", color="#e6edf3", fontsize=10)
            st.pyplot(fig_g); plt.close(fig_g)

        # ── Individual model cosine scores ───────────────────────────────
        st.markdown('<div class="section-title">Model-wise Cosine Similarity</div>',
                    unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        for col_w, label, val, color in [
            (mc1, "buffalo_l\n(cosine)", cos_l, "#0288d1"),
            (mc2, "buffalo_m\n(cosine)", cos_m, "#f0883e"),
            (mc3, "HybridFusionNet\n(sigmoid)", score, gc),
        ]:
            with col_w:
                fig_m, ax_m = plt.subplots(figsize=(2.5,2), facecolor="#0d1117")
                ax_m.set_facecolor("#161b22")
                ax_m.bar([label], [val], color=color, width=0.5, edgecolor="#30363d")
                ax_m.axhline(THRESHOLD, color="#ff8f00", linestyle="--", lw=1.5)
                ax_m.set_ylim(0,1.1)
                ax_m.set_title(f"{val:.4f}", color=color, fontsize=13,
                               fontweight="bold", fontfamily="monospace")
                ax_m.tick_params(colors="#8b949e", labelsize=7)
                for sp in ax_m.spines.values(): sp.set_color("#30363d")
                ax_m.yaxis.grid(True, color="#30363d", alpha=0.3)
                st.pyplot(fig_m); plt.close(fig_m)

        # ── Per-segment bar chart ─────────────────────────────────────────
        st.markdown('<div class="section-title">Feature Segment Similarity Breakdown</div>',
                    unsafe_allow_html=True)

        # Analyse each embedder's segments separately + full concat
        seg_l   = segment_cosine(emb_l_old, emb_l_new, n=8)   # buffalo_l segments
        seg_m   = segment_cosine(emb_m_old, emb_m_new, n=8)   # buffalo_m segments
        labels  = [f"L{i+1}" for i in range(8)] + [f"M{i+1}" for i in range(8)]
        all_seg = seg_l + seg_m
        b_colors = ["#0288d1" if s>=THRESHOLD else "#1a5276" for s in seg_l] + \
                   ["#f0883e" if s>=THRESHOLD else "#784212" for s in seg_m]

        fig_bar, ax_bar = plt.subplots(figsize=(12,4.5), facecolor="#0d1117")
        ax_bar.set_facecolor("#161b22")
        bars = ax_bar.bar(labels, all_seg, color=b_colors, edgecolor="#30363d",
                          linewidth=1.1, zorder=3, width=0.7)
        for b, v in zip(bars, all_seg):
            ax_bar.text(b.get_x()+b.get_width()/2, b.get_height()+0.013,
                        f"{v:.2f}", ha="center", va="bottom",
                        color="#e6edf3", fontsize=7.5, fontfamily="monospace")
        ax_bar.axhline(THRESHOLD, color="#ff8f00", linestyle="--", lw=2, zorder=4,
                       label=f"Threshold ({THRESHOLD})")
        ax_bar.axhline(score, color="#ef5350", linestyle="-.", lw=1.8, zorder=4,
                       label=f"Hybrid Score ({score:.3f})")
        ax_bar.axvline(7.5, color="#8b949e", lw=1.5, linestyle=":")   # split L/M
        ax_bar.text(3.5, 1.05, "buffalo_l segments",
                    ha="center", color="#0288d1", fontsize=9, fontweight="bold")
        ax_bar.text(11.5, 1.05, "buffalo_m segments",
                    ha="center", color="#f0883e", fontsize=9, fontweight="bold")
        ax_bar.axhspan(THRESHOLD, 1.07, alpha=0.05, color="#66bb6a")
        ax_bar.axhspan(0.0, THRESHOLD, alpha=0.05, color="#ef5350")
        ax_bar.set_ylim(0, 1.12)
        ax_bar.set_xlabel("Feature Segment", color="#8b949e", fontsize=10)
        ax_bar.set_ylabel("Cosine Similarity", color="#8b949e", fontsize=10)
        ax_bar.set_title("Per-Segment Feature Similarity — buffalo_l (L1–L8) + buffalo_m (M1–M8)",
                         color="#e6edf3", fontsize=12)
        ax_bar.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax_bar.spines.values(): sp.set_color("#30363d")
        ax_bar.yaxis.grid(True, color="#30363d", linestyle="--", alpha=0.35, zorder=0)
        ax_bar.legend(facecolor="#161b22", edgecolor="#30363d",
                      labelcolor="#e6edf3", fontsize=9, loc="upper right")
        st.pyplot(fig_bar); plt.close(fig_bar)

        # ── Embedding overlay ─────────────────────────────────────────────
        st.markdown('<div class="section-title">Embedding Vector Comparison</div>',
                    unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["buffalo_l (512-d)", "buffalo_m (512-d)"])
        for tab, e_old, e_new, label, color in [
            (tab1, emb_l_old, emb_l_new, "buffalo_l", "#0288d1"),
            (tab2, emb_m_old, emb_m_new, "buffalo_m", "#f0883e"),
        ]:
            with tab:
                xs = np.arange(len(e_old))
                fig_l, ax_l = plt.subplots(figsize=(11,3), facecolor="#0d1117")
                ax_l.set_facecolor("#161b22")
                ax_l.plot(xs, e_old, color=color, lw=1, alpha=.9, label="OLD")
                ax_l.plot(xs, e_new, color="#e6edf3", lw=1, alpha=.6, label="NEW")
                ax_l.fill_between(xs, e_old, e_new, alpha=.15, color="#ef5350",
                                  label="Difference")
                ax_l.set_title(f"{label} Embedding — OLD vs NEW",
                               color="#e6edf3", fontsize=11)
                ax_l.set_xlabel("Dimension", color="#8b949e")
                ax_l.set_ylabel("Activation", color="#8b949e")
                ax_l.tick_params(colors="#8b949e")
                for sp in ax_l.spines.values(): sp.set_color("#30363d")
                ax_l.legend(facecolor="#161b22", edgecolor="#30363d",
                            labelcolor="#e6edf3", fontsize=9)
                st.pyplot(fig_l); plt.close(fig_l)

        # ── Difference heatmap ────────────────────────────────────────────
        st.markdown('<div class="section-title">Embedding Difference Heatmap</div>',
                    unsafe_allow_html=True)

        diff_l = np.abs(emb_l_old - emb_l_new).reshape(1,-1)
        diff_m = np.abs(emb_m_old - emb_m_new).reshape(1,-1)
        diff   = np.vstack([diff_l, diff_m])

        fig_hm, ax_hm = plt.subplots(figsize=(12,1.8), facecolor="#0d1117")
        ax_hm.set_facecolor("#161b22")
        im = ax_hm.imshow(diff, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.3)
        plt.colorbar(im, ax=ax_hm, pad=0.01)
        ax_hm.set_yticks([0,1])
        ax_hm.set_yticklabels(["buffalo_l","buffalo_m"], color="#8b949e", fontsize=9)
        ax_hm.set_xlabel("Dimension", color="#8b949e", fontsize=9)
        ax_hm.set_title("|OLD − NEW| per dimension   (green = similar · red = different)",
                        color="#e6edf3", fontsize=10)
        ax_hm.tick_params(colors="#8b949e", labelsize=7)
        for sp in ax_hm.spines.values(): sp.set_color("#30363d")
        st.pyplot(fig_hm); plt.close(fig_hm)

        # ── Summary table ─────────────────────────────────────────────────
        st.markdown('<div class="section-title">Verification Summary</div>',
                    unsafe_allow_html=True)
        df = pd.DataFrame({
            "Metric": [
                "HybridFusionNet Score", "buffalo_l Cosine", "buffalo_m Cosine",
                "Decision Threshold", "Match Result",
                "buffalo_l Segments ≥ Threshold",
                "buffalo_m Segments ≥ Threshold",
            ],
            "Value": [
                f"{score:.6f}",
                f"{cos_l:.6f}",
                f"{cos_m:.6f}",
                str(THRESHOLD),
                "✅ MATCH" if match else "❌ MISMATCH",
                f"{sum(s>=THRESHOLD for s in seg_l)} / 8",
                f"{sum(s>=THRESHOLD for s in seg_m)} / 8",
            ]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center;color:#8b949e;font-size:.76rem;font-family:'IBM Plex Mono',monospace">
UIDAI Face Biometric &nbsp;|&nbsp; buffalo_l ⊕ buffalo_m → HybridFusionNet &nbsp;|&nbsp; Authorised use only
</p>""", unsafe_allow_html=True)