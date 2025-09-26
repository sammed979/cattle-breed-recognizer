from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
app = FastAPI()
MODEL_PATH = "../model/breed_predictor.tflite"
IMG_SIZE = (224, 224)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dummy breed labels (replace with actual)
breed_labels = [f"Breed_{i}" for i in range(74)]

def authenticate(token: str = Depends(oauth2_scheme)):
    # Dummy authentication, replace with real token validation
    if token != "securetoken":
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), token: str = Depends(authenticate)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = np.argmax(output)
    breed = breed_labels[pred_idx]
    confidence = float(np.max(output))
    return JSONResponse({"breed": breed, "confidence": confidence})

@app.post("/feedback")
async def submit_feedback(image_id: str = Form(...), correct_breed: str = Form(...), token: str = Depends(authenticate)):
    # Store feedback for retraining
    with open("feedback.csv", "a") as f:
        f.write(f"{image_id},{correct_breed}\n")
    return {"status": "Feedback received"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), breed: str = Form(...), species: str = Form(...), traits: str = Form(...), token: str = Depends(authenticate)):
    # Save user-uploaded image for dataset expansion
    save_dir = "user_uploads"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    with open("uploads.csv", "a") as f:
        f.write(f"{file.filename},{breed},{species},{traits}\n")
    return {"status": "Image uploaded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
