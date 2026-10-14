# LS-MILLS: Low-Light Face Recognition & Biometric Enhancement System

LS-MILLS is a biometric face verification and enhancement system designed for challenging lighting and degraded image conditions. It integrates deep feature extraction, super-resolution enhancement, hybrid matching, and real-time cloud database synchronization with Supabase.

---

## 🚀 Key Features

- **Biometric Face Verification & Enrollment**: Multi-model face recognition and verification pipeline with high precision.
- **Image Enhancement (Real-ESRGAN)**: Automatic low-light enhancement and resolution restoration for degraded face captures.
- **Hybrid Matching Architecture**: Blends deep embeddings for robust identity matching across varying angles, resolutions, and illumination.
- **Cloud Database (Supabase)**: Real-time synchronization of enrolled members and verification audit logs.
- **Interactive Streamlit UI**: User-friendly web dashboard featuring live camera capture, manual upload, enrollment management, and audit verification history.

---

## 📁 Project Structure

```text
├── api.py                    # FastAPI server for inference endpoints
├── dataset.py                # Dataset loading and preprocessing utilities
├── enhancement.py            # Super-resolution (Real-ESRGAN) pipeline
├── models.py                 # Face recognition model definitions & embeddings
├── requirements.txt          # Python dependencies
├── streamlit_app.py          # Primary Streamlit web application
├── streamlit_app_light.py    # Streamlit UI with Supabase cloud integration
├── supabase_db.py            # Supabase database operations & sync manager
├── train.py                  # Training pipeline for hybrid verification
├── hybrid_model/             # Precomputed embeddings & trained hybrid weights
├── weights/                  # Directory for RealESRGAN weights (auto-downloaded)
├── test1/                    # Benchmarks, metrics, and visualization curves
└── result/                   # Local enrolled identity database backup
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/MADHURESH198181141/LS-MILLS.git
cd LS-MILLS
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
SUPABASE_URL="https://your-supabase-project.supabase.co"
SUPABASE_KEY="your-supabase-api-key"
```

---

## 💻 Running the Application

### Launch Streamlit Dashboard
```bash
streamlit run streamlit_app_light.py
```

### Run FastAPI Service
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📊 Evaluation & Metrics
Performance benchmarks, DET/ROC curves, and comparative metrics are located in the `test1/` directory.
