from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from inference_module import run_inference  # Your inference function

app = FastAPI()

# CORS will be configured later (see step 5.4)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # Process contents (e.g., convert to image, keypoints, etc.)
    # For simplicity, assume run_inference handles the conversion.
    prediction = run_inference(contents)
    return {"gesture": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
