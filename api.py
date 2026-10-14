from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os

# --- IMPORT YOUR PROJECT CODE HERE ---
# from your_project import extract_embedding, calculate_cosine_similarity

app = FastAPI(title="Face Verification API")

# Path to the reference Aadhaar image already on your PC
REFERENCE_IMAGE_PATH = "reference_aadhaar.jpg" 

@app.post("/verify-selfie/")
async def verify_selfie(selfie: UploadFile = File(...)):
    # 1. Save the incoming selfie from Telegram
    selfie_path = f"temp_{selfie.filename}"
    with open(selfie_path, "wb") as buffer:
        shutil.copyfileobj(selfie.file, buffer)

    # 2. RUN YOUR AI PIPELINE (Real-ESRGAN -> CodeFormer -> SCRFD -> IResNet50)
    try:
        # embedding_reference = extract_embedding(REFERENCE_IMAGE_PATH)
        # embedding_selfie = extract_embedding(selfie_path)
        # similarity_score = calculate_cosine_similarity(embedding_reference, embedding_selfie)
        
        # --- SIMULATED RESULT (Replace with your actual function outputs) ---
        similarity_score = 0.65  # Your threshold is 0.4 - 0.6
        threshold = 0.5
        is_match = similarity_score >= threshold

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Clean up the temporary selfie file
        if os.path.exists(selfie_path):
            os.remove(selfie_path)

    # 3. Return results to n8n
    return JSONResponse(content={
        "match": bool(is_match),
        "similarity_score": float(similarity_score),
        "message": "Identity Verified" if is_match else "Verification Failed"
    })

if __name__ == "__main__":
    # Runs the API on your local PC
    uvicorn.run(app, host="0.0.0.0", port=8000)