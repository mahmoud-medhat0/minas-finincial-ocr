import sys
import os
from pathlib import Path

# Add parent directory to sys.path to import financial_ocr
# This mimics main.py lines 14-15
sys.path.append(str(Path(__file__).parent.parent))

from financial_ocr import process_document

# Configure paths exactly as main.py does
# Line 33 of main.py
TEMPLATES_PATH = os.path.join(os.path.dirname(__file__), "..", "templates.json")
TESSERACT_PATH = r"D:\tesseract\tesseract.exe"

file_path = r"C:/Users/NEGM/.gemini/antigravity/brain/41038a5c-eb3c-4baa-9adc-07c4113f2839/uploaded_image_1768792104835.png"

print(f"Using templates from: {os.path.abspath(TEMPLATES_PATH)}")
print(f"Templates file exists: {os.path.exists(TEMPLATES_PATH)}")

try:
    result = process_document(file_path, TEMPLATES_PATH, tesseract_cmd=TESSERACT_PATH)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
