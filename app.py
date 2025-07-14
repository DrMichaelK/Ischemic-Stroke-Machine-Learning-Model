#app.py
import os
import io
import uvicorn
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchvision import models, transforms
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import pydicom
import gc
from model import CombinedModel, ImageToTextProjector
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root(request: Request):
    return {"message": "Welcome to Phronesis"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dicom_to_png(dicom_data):
    try:
        dicom_file = pydicom.dcmread(dicom_data)
        if not hasattr(dicom_file, 'PixelData'):
            raise HTTPException(status_code=400, detail="No pixel data in DICOM file.")
        
        pixel_array = dicom_file.pixel_array.astype(np.float32)
        pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.ptp())) * 255.0
        pixel_array = pixel_array.astype(np.uint8)

        img = Image.fromarray(pixel_array).convert("L")
        return img
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting DICOM to PNG: {e}")

# Set up secure model initialization
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("Missing Hugging Face token in environment variables.")

try:
    report_generator_tokenizer = AutoTokenizer.from_pretrained(
        "KYAGABA/combined-multimodal-model",
        token=HF_TOKEN if HF_TOKEN else None
    )
    video_model = models.video.r3d_18(weights="KINETICS400_V1")
    video_model.fc = torch.nn.Linear(video_model.fc.in_features, 512)
    report_generator = AutoModelForSeq2SeqLM.from_pretrained("GanjinZero/biobart-v2-base")
    projector = ImageToTextProjector(512, report_generator.config.d_model)
    num_classes = 4
    combined_model = CombinedModel(video_model, report_generator, num_classes, projector, report_generator_tokenizer)
    model_file = hf_hub_download("KYAGABA/combined-multimodal-model", "pytorch_model.bin", token=HF_TOKEN)
    state_dict = torch.load(model_file, map_location=device)
    combined_model.load_state_dict(state_dict)
    
    # Move model to device
    combined_model = combined_model.to(device)
    combined_model.eval()
    print("Models loaded successfully!")
except Exception as e:
    raise SystemExit(f"Error loading models: {e}")

image_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

class_names = ["acute", "normal", "chronic", "lacunar"]

@app.post("/predict/")
async def predict(files: list[UploadFile]):
    try:
        print(f"Received {len(files)} files")
        n_frames = 16
        images = []

        for file in files:
            ext = file.filename.split('.')[-1].lower()
            try:
                if ext in ['dcm', 'ima']:
                    dicom_img = dicom_to_png(await file.read())
                    images.append(dicom_img.convert("RGB"))
                elif ext in ['png', 'jpeg', 'jpg']:
                    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
                    images.append(img)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file type.")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {e}")

        if not images:
            return JSONResponse(content={"error": "No valid images provided."}, status_code=400)

        if len(images) >= n_frames:
            images_sampled = [images[i] for i in np.linspace(0, len(images) - 1, n_frames, dtype=int)]
        else:
            images_sampled = images + [images[-1]] * (n_frames - len(images))

        image_tensors = [image_transform(img) for img in images_sampled]
        images_tensor = torch.stack(image_tensors).permute(1, 0, 2, 3).unsqueeze(0)
        
        # Ensure tensor is on the same device as model
        images_tensor = images_tensor.to(device)

        with torch.no_grad():
            class_outputs, generated_report, _ = combined_model(images_tensor)
            predicted_class = torch.argmax(class_outputs, dim=1).item()
            predicted_class_name = class_names[predicted_class]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {
            "predicted_class": predicted_class_name,
            "generated_report": generated_report[0] if generated_report else "No report generated."
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during prediction: {str(e)}\n{error_details}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)