from financial_ocr import process_document
import json
import sys

try:
    # Use the hardcoded path from financial_ocr.py or pass it
    tesseract_path = r"D:\tesseract\tesseract.exe"
    
    file_path = r"C:/Users/NEGM/.gemini/antigravity/brain/41038a5c-eb3c-4baa-9adc-07c4113f2839/uploaded_image_1768791867837.png"
    result = process_document(file_path, "templates.json", tesseract_cmd=tesseract_path)
    
    with open("snb_extracted.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print("Optimization successful. Written to snb_extracted.json")
except Exception as e:
    print(f"Error: {e}")
