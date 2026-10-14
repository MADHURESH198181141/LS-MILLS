"""
UIDAI Face Matching — Streamlit Testing App
Tests the complete pipeline:
  Enroll Flow: Quality Check → ESRGAN 2x → CodeFormer → SCRFD → GlintR100 → Save
  Match Flow:  Quality Check → ESRGAN 2x → SCRFD → GlintR100 → Cosine Similarity
  Classify:    On-the-spot identification against all enrolled faces
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
import streamlit as st
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
_candidate_fgnet = os.path.join(_PARENT_DIR, "FGNET (1)")
if os.path.exists(_candidate_fgnet):
    BASE_DIR = _candidate_fgnet
elif os.path.exists(os.path.join(_SCRIPT_DIR, "FGNET (1)")):
    BASE_DIR = os.path.join(_SCRIPT_DIR, "FGNET (1)")
else:
    BASE_DIR = _SCRIPT_DIR

RESULT_DIR = os.path.join(BASE_DIR, "result")
MODEL_PATH = os.path.join(RESULT_DIR, "face_recognition_int8.onnx")
DB_PATH    = os.path.join(RESULT_DIR, "enrolled_db.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enhancement import EnhancementPipeline, needs_enhancement, laplacian_quality_score
from models import InsightFaceRecognizer, cosine_similarity
import supabase_db


# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="UIDAI Face Matching",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    
    .main { 
        background: linear-gradient(135deg, #090b14 0%, #121526 100%);
    }
    .stApp { 
        background-color: transparent;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(14, 17, 28, 0.7) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(22, 26, 40, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        margin: 12px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 40px 0 rgba(79, 195, 247, 0.15);
        border-color: rgba(79, 195, 247, 0.4);
    }
    .metric-value { 
        font-size: 2.8rem; 
        font-family: 'Outfit', sans-serif;
        font-weight: 800; 
        background: -webkit-linear-gradient(45deg, #4fc3f7, #00e5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label { 
        font-size: 0.9rem; 
        color: #90a4ae; 
        margin-top: 8px; 
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Result Cards */
    .match-success {
        background: linear-gradient(135deg, rgba(27, 94, 32, 0.75), rgba(46, 125, 50, 0.75));
        border: 1px solid #4caf50;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        font-size: 1.8rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        color: #ccff90;
        box-shadow: 0 0 25px rgba(76, 175, 80, 0.3);
        animation: pulse-green 2s infinite;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .match-fail {
        background: linear-gradient(135deg, rgba(183, 28, 28, 0.75), rgba(198, 40, 40, 0.75));
        border: 1px solid #ef5350;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        font-size: 1.8rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        color: #ffcdd2;
        box-shadow: 0 0 25px rgba(239, 83, 80, 0.3);
        animation: pulse-red 2s infinite;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    /* Pipeline Steps */
    .pipeline-step {
        background: rgba(22, 26, 40, 0.7);
        border-left: 5px solid #4fc3f7;
        padding: 16px 20px;
        margin: 10px 0;
        border-radius: 0 12px 12px 0;
        font-size: 0.95rem;
        color: #eceff1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    .pipeline-step:hover {
        background: rgba(30, 35, 53, 0.9);
        transform: translateX(8px);
        border-left-color: #00e5ff;
        box-shadow: -2px 8px 20px rgba(0,229,255,0.15);
    }
    .pipeline-skip {
        border-left-color: #607d8b;
        opacity: 0.6;
    }
    .pipeline-skip:hover {
        border-left-color: #90a4ae;
        box-shadow: none;
    }
    
    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0288d1 0%, #01579b 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        font-family: 'Outfit', sans-serif;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px rgba(2, 136, 209, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #03a9f4 0%, #0288d1 100%);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(3, 169, 244, 0.4);
        border-color: rgba(255,255,255,0.3);
    }
    .stButton > button:active {
        transform: translateY(1px) scale(0.98);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        border-radius: 20px;
        border: 2px dashed rgba(255,255,255,0.15);
        padding: 24px;
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #4fc3f7;
        background: rgba(79, 195, 247, 0.05);
        box-shadow: 0 0 20px rgba(79, 195, 247, 0.1);
    }
    
    /* Input fields */
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background-color: rgba(14, 17, 28, 0.6) !important;
        color: white !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus, .stSelectbox > div > div > div:focus {
        border-color: #4fc3f7 !important;
        box-shadow: 0 0 0 2px rgba(79, 195, 247, 0.2) !important;
        background-color: rgba(14, 17, 28, 0.9) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(22, 26, 40, 0.4);
        border-radius: 16px 16px 0 0;
        border: 1px solid rgba(255,255,255,0.05);
        border-bottom: none;
        padding: 12px 28px;
        color: #90a4ae;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(30, 35, 53, 0.6);
        color: #e3f2fd;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(30, 35, 53, 0.9) 0%, rgba(14, 17, 28, 0) 100%);
        color: #4fc3f7 !important;
        border-color: rgba(79, 195, 247, 0.3);
        border-bottom: 2px solid #4fc3f7;
    }

    /* Streamlit Metric Overrides */
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        color: #e3f2fd !important;
    }

    /* Keyframes */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 83, 80, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(239, 83, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 83, 80, 0); }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "enrolled_db" not in st.session_state:
    st.session_state.enrolled_db = {}
    st.session_state.db_mode = "local"
    try:
        if supabase_db.is_connected():
            st.session_state.enrolled_db = supabase_db.fetch_all_enrolled()
            st.session_state.db_mode = "supabase"
    except Exception as e:
        print(f"[Supabase] Connect error: {e}")

    # Fallback to local file if not loaded from Supabase and local exists
    if not st.session_state.enrolled_db and os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r") as f:
                raw = json.load(f)
                for k, v in raw.items():
                    if isinstance(v, dict) and "embedding" in v:
                        st.session_state.enrolled_db[k] = {
                            "name": v.get("name", "Unknown"),
                            "embedding": np.array(v["embedding"]),
                            "history": v.get("history", []),
                            "image_url": v.get("image_url")
                        }
                    else:
                        st.session_state.enrolled_db[k] = {
                            "name": f"User {k}",
                            "embedding": np.array(v),
                            "history": [],
                            "image_url": None
                        }
        except Exception:
            pass

if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []


# ─────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load all models once and cache them."""
    with st.spinner("Loading AI models... please wait"):
        device = "cpu"  # CPU for UIDAI testing

        # Enhancement pipeline
        enhancer = EnhancementPipeline(device=device, quality_threshold=100.0)

        # InsightFace recognizer
        try:
            recognizer = InsightFaceRecognizer(model_name="antelopev2", device=device)
        except Exception:
            recognizer = InsightFaceRecognizer(model_name="buffalo_l", device=device)

    return enhancer, recognizer


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def save_member_record(aadhaar_id, name, embedding, image_bytes=None):
    """Save enrolled member to Supabase and keep local JSON backup."""
    saved_cloud = False
    try:
        saved_cloud = supabase_db.save_enrolled_member(aadhaar_id, name, embedding, image_bytes=image_bytes)
    except Exception as e:
        print(f"[Supabase] Save error: {e}")
    save_local_backup()
    return bool(saved_cloud)


def log_verification_record(aadhaar_id, match_rate_str, status_str, similarity_score=0.0):
    """Log verification attempt to Supabase and local backup."""
    try:
        supabase_db.log_verification(aadhaar_id, match_rate_str, status_str, similarity_score)
    except Exception as e:
        print(f"[Supabase] Log error: {e}")
    save_local_backup()

def clear_database():
    """Clear all records from Supabase cloud and local file."""
    st.session_state.enrolled_db = {}
    try:
        supabase_db.clear_all_enrolled()
    except Exception as e:
        print(f"[Supabase] Clear error: {e}")
    save_local_backup()

def save_local_backup():
    """Save current session state to local JSON file as backup."""
    try:
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(DB_PATH, "w") as f:
            converted = {}
            for k, v in st.session_state.enrolled_db.items():
                converted[k] = {
                    "name": v["name"],
                    "embedding": v["embedding"].tolist() if isinstance(v["embedding"], np.ndarray) else list(v["embedding"]),
                    "history": v.get("history", []),
                    "image_url": v.get("image_url")
                }
            json.dump(converted, f)
    except Exception:
        pass



def run_enroll_pipeline(image_cv2, enhancer, recognizer):
    """Full enroll pipeline with step logging."""
    steps = []
    t_start = time.time()

    quality = laplacian_quality_score(image_cv2)
    poor = needs_enhancement(image_cv2)
    steps.append({"step": "Quality Check", "result": f"Score: {quality:.1f} → {'Enhancing' if poor else 'Skip'}", "skipped": False})

    if poor:
        t = time.time()
        enhanced = enhancer.esrgan.enhance(image_cv2)
        steps.append({"step": "Real-ESRGAN 2x", "result": f"Done in {time.time()-t:.2f}s", "skipped": False})
        
        t = time.time()
        restored = enhancer.codeformer.enhance(enhanced)
        steps.append({"step": "CodeFormer", "result": f"Done in {time.time()-t:.2f}s", "skipped": False})
    else:
        restored = image_cv2
        steps.append({"step": "Real-ESRGAN 2x", "result": "Skipped", "skipped": True})
        steps.append({"step": "CodeFormer", "result": "Skipped", "skipped": True})

    t = time.time()
    embedding = recognizer.get_embedding(restored)
    if embedding is None:
        steps.append({"step": "SCRFD + GlintR100", "result": "❌ No face detected!", "skipped": False})
        return None, steps, time.time() - t_start

    steps.append({"step": "SCRFD + GlintR100 Embedding", "result": f"Done in {time.time()-t:.2f}s", "skipped": False})
    return embedding, steps, time.time() - t_start


def run_match_pipeline(image_cv2, enrolled_emb, enhancer, recognizer):
    """Match pipeline — CodeFormer skipped for speed."""
    steps = []
    t_start = time.time()

    quality = laplacian_quality_score(image_cv2)
    poor = needs_enhancement(image_cv2)
    steps.append({"step": "Quality Check", "result": f"Score: {quality:.1f} → {'Enhancing' if poor else 'Skip'}", "skipped": False})

    if poor:
        t = time.time()
        enhanced = enhancer.esrgan.enhance(image_cv2)
        steps.append({"step": "Real-ESRGAN 2x", "result": f"Done in {time.time()-t:.2f}s", "skipped": False})
    else:
        enhanced = image_cv2
        steps.append({"step": "Real-ESRGAN 2x", "result": "Skipped", "skipped": True})

    steps.append({"step": "CodeFormer", "result": "⚡ Skipped (faster)", "skipped": True})

    t = time.time()
    query_emb = recognizer.get_embedding(enhanced)
    if query_emb is None:
        steps.append({"step": "SCRFD + GlintR100", "result": "❌ No face detected!", "skipped": False})
        return None, steps, time.time() - t_start

    steps.append({"step": "SCRFD + GlintR100 Embedding", "result": f"Done in {time.time()-t:.2f}s", "skipped": False})

    score = cosine_similarity(query_emb, enrolled_emb)
    matched = score >= 0.3
    percentage = max(0.0, min(100.0, score * 100))
    
    steps.append({"step": "Cosine Similarity", "result": f"Match Rate: {percentage:.1f}% | {'MATCH ✅' if matched else 'NO MATCH ❌'}", "skipped": False})

    return {"score": score, "percentage": percentage, "matched": matched}, steps, time.time() - t_start


def display_pipeline_steps(steps):
    st.markdown("#### 🔄 Pipeline Execution")
    for s in steps:
        css = "pipeline-step pipeline-skip" if s["skipped"] else "pipeline-step"
        icon = "⏭️" if s["skipped"] else "✅"
        st.markdown(f'<div class="{css}">{icon} <b>{s["step"]}</b> — {s["result"]}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN APP WITH SIDEBAR NAVIGATION
# ─────────────────────────────────────────

def main():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/200px-Aadhaar_Logo.svg.png", width=120)
    st.sidebar.title("UIDAI Portal")
    st.sidebar.markdown("---")
    
    # Navigation Menu
    page = st.sidebar.radio(
        "Navigation", 
        ["📥 Enroll Person", "🔍 Verify & Match", "📊 Members & History Dashboard", "📈 Training Metrics"]
    )
    
    st.sidebar.markdown("---")
    count = len(st.session_state.enrolled_db)
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{count}</div>
        <div class="metric-label">Enrolled DB</div>
    </div>
    """, unsafe_allow_html=True)

    # Cloud DB Connection status badge & sync
    is_cloud_active = supabase_db.is_connected()
    if is_cloud_active:
        st.session_state.db_mode = "supabase"
        st.sidebar.markdown("""
        <div style="background: rgba(0, 230, 118, 0.1); border: 1px solid #00e676; border-radius: 12px; padding: 10px; text-align: center; margin-bottom: 10px;">
            <span style="color: #00e676; font-weight: 700; font-size: 0.85rem;">☁️ Supabase Cloud DB Connected</span>
        </div>
        """, unsafe_allow_html=True)
        col_s1, col_s2 = st.sidebar.columns(2)
        with col_s1:
            if st.button("🔄 Pull Cloud", key="sync_btn", use_container_width=True):
                with st.spinner("Fetching from Supabase..."):
                    cloud_records = supabase_db.fetch_all_enrolled()
                    if cloud_records:
                        st.session_state.enrolled_db = cloud_records
                        save_local_backup()
                    st.success("Synced!")
                    st.rerun()
        with col_s2:
            if st.button("⬆️ Push Local", key="push_btn", use_container_width=True):
                with st.spinner("Pushing to Supabase..."):
                    ok_cnt, fail_cnt = supabase_db.sync_local_to_supabase(st.session_state.enrolled_db)
                    st.success(f"Pushed {ok_cnt} records!")
                    st.rerun()
    else:
        st.sidebar.markdown("""
        <div style="background: rgba(255, 179, 0, 0.1); border: 1px solid #ffb300; border-radius: 12px; padding: 10px; text-align: center; margin-bottom: 10px;">
            <span style="color: #ffb300; font-weight: 700; font-size: 0.85rem;">📁 Local JSON Mode (Cloud Offline)</span>
        </div>
        """, unsafe_allow_html=True)
        if st.sidebar.button("🔌 Reconnect Supabase", key="retry_supa_btn", use_container_width=True):
            if supabase_db.is_connected():
                st.session_state.db_mode = "supabase"
                st.session_state.enrolled_db = supabase_db.fetch_all_enrolled()
                st.sidebar.success("Connected to Supabase!")
                st.rerun()
            else:
                st.sidebar.error("Could not connect to Supabase.")

    if st.sidebar.button("🗑️ Clear Database"):
        clear_database()
        st.sidebar.success("Database Cleared from Supabase & Local!")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Verification")
    quick_aadhaar = st.sidebar.text_input("Aadhaar ID", key="quick_aadhaar", placeholder="1234-5678-9012")
    quick_upload = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="quick_upload")
    if st.sidebar.button("Run Quick Verify", key="quick_btn"):
        if not quick_aadhaar:
            st.sidebar.error("Enter Aadhaar ID")
        elif quick_aadhaar not in st.session_state.enrolled_db:
            st.sidebar.error("❌ Not found")
        elif not quick_upload:
            st.sidebar.error("Upload image")
        else:
            with st.sidebar.status("Running verification...", expanded=True) as status:
                try:
                    enh_q, rec_q = load_models()
                    q_user = st.session_state.enrolled_db[quick_aadhaar]
                    q_img = pil_to_cv2(Image.open(quick_upload).convert("RGB"))
                    res, stp, el = run_match_pipeline(q_img, q_user["embedding"], enh_q, rec_q)
                    
                    if res:
                        pct = res["percentage"]
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        stat_text = "Success ✅" if res["matched"] else "Failed ❌"
                        q_user["history"].insert(0, {
                            "timestamp": timestamp,
                            "match_rate": f"{pct:.1f}%",
                            "status": stat_text
                        })
                        log_verification_record(quick_aadhaar, f"{pct:.1f}%", stat_text, similarity_score=res.get("score", 0.0))

                        
                        if res["matched"]:
                            st.sidebar.success(f"✅ MATCH CONFIRMED! {pct:.1f}%")
                            status.update(label="Match Verified", state="complete")
                        else:
                            st.sidebar.error(f"❌ MISMATCH! {pct:.1f}%")
                            status.update(label="Verification Failed", state="error")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
                    status.update(label="Error Occurred", state="error")

    # Title Area
    st.title("🔐 UIDAI Biometric System")
    st.markdown("*Age-Variation Face Verification Challenge — SDK Portal*")
    st.markdown("---")

    # Load models
    try:
        enhancer, recognizer = load_models()
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()


    # ── PAGE 1: ENROLL ────────────────────────────────────────────
    if page == "📥 Enroll Person":
        st.header("📥 Enroll New Identity")
        st.markdown("Add a new member to the UIDAI secure database.")

        col1, col2 = st.columns([1, 1])

        with col1:
            name_input = st.text_input("Full Name", placeholder="e.g. John Doe")
            aadhaar_input = st.text_input("Aadhaar ID", placeholder="e.g. 1234-5678-9012")
            uploaded = st.file_uploader("Upload Enrollment Image", type=["jpg", "jpeg", "png"], key="enroll_upload")

            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                st.image(pil_img, caption="Source Image", use_column_width=True)

            if st.button("Enroll Member", key="enroll_btn"):
                if not name_input or not aadhaar_input:
                    st.error("Please provide both Name and Aadhaar ID.")
                elif not uploaded:
                    st.error("Please upload an image.")
                elif aadhaar_input in st.session_state.enrolled_db:
                    st.warning("⚠️ This Aadhaar ID is already enrolled! Please clear or use a new ID.")
                else:
                    with st.spinner("Extracting embeddings & enrolling..."):
                        img_cv2 = pil_to_cv2(pil_img)
                        embedding, steps, elapsed = run_enroll_pipeline(img_cv2, enhancer, recognizer)

                    with col2:
                        display_pipeline_steps(steps)

                        if embedding is not None:
                            # Save to Database Schema
                            st.session_state.enrolled_db[aadhaar_input] = {
                                "name": name_input,
                                "embedding": embedding,
                                "history": []
                            }
                            img_bytes = uploaded.getvalue() if uploaded else None
                            cloud_saved = save_member_record(aadhaar_input, name_input, embedding, image_bytes=img_bytes)

                            if cloud_saved:
                                st.markdown(f"""
                                <div class="match-success">
                                    ✅ {name_input} Enrolled Successfully!<br>
                                    <small>Aadhaar: {aadhaar_input} | ☁️ Saved to Supabase Cloud & Local Storage</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="match-success" style="border-color: #ffb300; color: #ffb300; background: rgba(255, 179, 0, 0.1);">
                                    ⚠️ {name_input} Enrolled in Local Backup!<br>
                                    <small>Aadhaar: {aadhaar_input} | Saved to local backup (Supabase sync failed or offline)</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error("❌ No face detected in the image.")


    # ── PAGE 2: VERIFY & MATCH ────────────────────────────────────
    elif page == "🔍 Verify & Match":
        st.header("🔍 1:1 Identity Verification")
        st.markdown("Authenticate a person against their stored Aadhaar embeddings.")

        col1, col2 = st.columns([1, 1])

        with col1:
            match_aadhaar = st.text_input("Aadhaar ID to Verify", placeholder="Enter Aadhaar ID")
            match_upload = st.file_uploader("Upload Live Match Image", type=["jpg", "jpeg", "png"], key="match_upload")

            if match_upload:
                pil_match = Image.open(match_upload).convert("RGB")
                st.image(pil_match, caption="Live Query Image", use_column_width=True)

            if st.button("Run Verification", key="match_btn"):
                if not match_aadhaar:
                    st.error("Please enter an Aadhaar ID.")
                elif match_aadhaar not in st.session_state.enrolled_db:
                    st.error("❌ Aadhaar ID not found in database.")
                elif not match_upload:
                    st.error("Please upload a match image.")
                else:
                    target_user = st.session_state.enrolled_db[match_aadhaar]
                    with st.spinner(f"Verifying against {target_user['name']}..."):
                        img_cv2 = pil_to_cv2(pil_match)
                        enrolled_emb = target_user["embedding"]
                        result, steps, elapsed = run_match_pipeline(img_cv2, enrolled_emb, enhancer, recognizer)

                    with col2:
                        display_pipeline_steps(steps)

                        if result:
                            pct = result["percentage"]
                            matched = result["matched"]

                            # Log to history
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            stat_text = "Success ✅" if matched else "Failed ❌"
                            target_user["history"].insert(0, {
                                "timestamp": timestamp,
                                "match_rate": f"{pct:.1f}%",
                                "status": stat_text
                            })
                            log_verification_record(match_aadhaar, f"{pct:.1f}%", stat_text, similarity_score=result.get("score", 0.0))


                            if matched:
                                st.markdown(f"""
                                <div class="match-success">
                                    ✅ MATCH CONFIRMED<br>
                                    <small>{target_user['name']} | Match Rate: {pct:.1f}%</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="match-fail">
                                    ❌ VERIFICATION FAILED<br>
                                    <small>Identity Mismatch | Match Rate: {pct:.1f}%</small>
                                </div>
                                """, unsafe_allow_html=True)


    # ── PAGE 3: MEMBERS & HISTORY DASHBOARD ───────────────────────
    elif page == "📊 Members & History Dashboard":
        st.header("📊 Members & History Dashboard")
        st.markdown("Overview of all enrolled members and their authentication history logs.")

        db_keys = list(st.session_state.enrolled_db.keys())
        
        if not db_keys:
            st.info("No members enrolled yet. Go to the Enroll page to add users.")
        else:
            # 1. Global Member Table
            st.markdown("### 👥 Enrolled Members Registry")
            table_data = []
            for aid, v in st.session_state.enrolled_db.items():
                last_rate = "N/A"
                if v["history"]:
                    last_rate = v["history"][0]["match_rate"]
                
                table_data.append({
                    "Name": v["name"],
                    "Aadhaar ID": aid,
                    "Total Attempts": len(v["history"]),
                    "Latest Match %": last_rate
                })
            
            import pandas as pd
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            st.markdown("---")

            # 2. Individual History Lookup
            st.markdown("### 🔎 Individual History Lookup")
            lookup_id = st.selectbox("Select Aadhaar ID to view history:", db_keys)
            
            if lookup_id:
                user_data = st.session_state.enrolled_db[lookup_id]
                st.markdown(f"**Member Logs for:** `{user_data['name']}` (Aadhaar: *{lookup_id}*)")
                
                history = user_data["history"]
                if not history:
                    st.warning("No verification attempts logged for this user yet.")
                else:
                    st.dataframe(pd.DataFrame(history), use_container_width=True)


    # ── PAGE 4: METRICS ───────────────────────────────────────────
    elif page == "📈 Training Metrics":
        st.header("📈 Training Results & Metrics")

        metrics_files = sorted(Path(RESULT_DIR).glob("metrics_epoch_*.json")) if os.path.exists(RESULT_DIR) else []

        if not metrics_files:
            st.info("No training results found. Run train.py first to generate SDK benchmarking data.")
        else:
            latest = json.load(open(metrics_files[-1]))
            st.markdown("### 🏆 Latest Evaluation Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("EER", f"{latest.get('EER', 0):.4f}")
            c2.metric("FMR10000", f"{latest.get('FMR10000_FNMR', 0):.4f}")
            c3.metric("FMR1000", f"{latest.get('FMR1000_FNMR', 0):.4f}")
            c4.metric("ROC AUC", f"{latest.get('ROC_AUC', 0):.4f}")

            # Show ROC/DET plots
            roc_plots = sorted(Path(RESULT_DIR).glob("roc_det_epoch_*.png"))
            if roc_plots:
                st.markdown("### 📊 ROC & DET Curves")
                st.image(str(roc_plots[-1]), caption="Latest ROC & DET Curve", use_column_width=True)

            st.markdown("### 📋 All Epoch Metrics")
            import pandas as pd
            all_metrics = [{"Epoch": str(f).split("epoch_")[1].replace(".json", ""), **json.load(open(f))} for f in metrics_files]
            st.dataframe(pd.DataFrame(all_metrics), use_container_width=True)


if __name__ == "__main__":
    main()
