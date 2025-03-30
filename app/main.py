from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from inference_module import run_inference  # Your inference function

app = FastAPI()

# Configure CORS
# Replace "https://yourapp.netlify.app" with your actual Netlify URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://signspeak-trial.netlify.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # For simplicity, assume run_inference handles all data processing:
    prediction = run_inference(contents)
    return {"gesture": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
