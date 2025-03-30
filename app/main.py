from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.inference_module import run_inference  # Fixed import path for deployment
import os

app = FastAPI()

# Configure CORS for Netlify frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://signspeak-trial.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # If run_inference handles raw bytes or image decoding internally
    prediction = run_inference(contents)
    return {"gesture": prediction}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # fallback to 8000 for local dev
    uvicorn.run(app, host="0.0.0.0", port=port)
