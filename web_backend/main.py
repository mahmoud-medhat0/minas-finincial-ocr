import os
import shutil
import uuid
import logging
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
from pathlib import Path

# Add parent directory to sys.path to import financial_ocr
sys.path.append(str(Path(__file__).parent.parent))

from financial_ocr import process_document

app = FastAPI(title="Financial OCR API")

# Configure Tesseract path
TESSERACT_PATH = r"D:\tesseract\tesseract.exe"

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
TEMPLATES_PATH = os.path.join(os.path.dirname(__file__), "..", "templates.json")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class OCRResult(BaseModel):
    template: str
    confidence: str
    amount: Optional[float]
    currency: Optional[str]
    transaction_id: Optional[str]
    date: Optional[str]
    sender: Optional[str]
    receiver: Optional[str]
    iban: Optional[str]
    raw_text: str
    evidence: dict

@app.get("/templates")
async def get_templates():
    """Returns a list of available templates."""
    try:
        with open(TEMPLATES_PATH, 'r', encoding='utf-8') as f:
            templates = json.load(f)
            return [t["name"] for t in templates]
    except Exception as e:
        logging.error(f"Error loading templates: {e}")
        return []

@app.post("/process", response_model=OCRResult)
async def process_upload(file: UploadFile = File(...), template: Optional[str] = None):
    # Generate unique filename to avoid collisions
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        result = process_document(file_path, TEMPLATES_PATH, tesseract_cmd=TESSERACT_PATH, template_name=template)
        
        # Cleanup uploaded file
        os.remove(file_path)
        
        return result
    except Exception as e:
        logging.error(f"Error processing upload: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
