
import os
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(r"d:\work\newocr")))

from financial_ocr import process_document

def test_latest_image():
    image_path = r"C:\Users\NEGM\.gemini\antigravity\brain\0904692c-b382-4b6e-abdc-a93e7c53fafa\uploaded_image_1768788675127.png"
    templates_path = r"d:\work\newocr\templates.json"
    tesseract_cmd = r"D:\tesseract\tesseract.exe"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Processing image: {image_path}")
    try:
        result = process_document(image_path, templates_path, tesseract_cmd=tesseract_cmd)
        
        # Print results neatly
        print("\n--- RESULTS ---")
        for k, v in result.items():
            if k not in ['raw_text', 'evidence']:
                print(f"{k}: {v}")
        
        print("\n--- EVIDENCE ---")
        for k, v in result['evidence'].items():
            print(f"{k}: {v}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_latest_image()
