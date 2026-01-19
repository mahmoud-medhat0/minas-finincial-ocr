
import os
import sys
from pathlib import Path
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np

# Add parent directory to sys.path
sys.path.append(str(Path(r"d:\work\newocr")))

def test_preprocessing():
    image_path = r"C:\Users\NEGM\.gemini\antigravity\brain\0904692c-b382-4b6e-abdc-a93e7c53fafa\uploaded_image_1768789012827.png"
    tesseract_cmd = r"D:\tesseract\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    if not os.path.exists(image_path):
        print("Image not found")
        return

    print("--- Original ---")
    img = Image.open(image_path)
    
    # Method 1: Current (Grayscale + Autocontrast + Sharpen)
    img1 = img.convert('L')
    img1 = ImageOps.autocontrast(img1)
    img1 = img1.filter(ImageFilter.SHARPEN)
    text1 = pytesseract.image_to_string(img1, lang='ara+eng', config='--psm 3')
    print("Method 1 (Current):")
    print(text1)
    print("Contains 'els':", "els" in text1)
    print("Contains 'نجاح':", "نجاح" in text1)

    # Method 2: Binary Thresholding (Otsu)
    # Convert PIL to CV2
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img2 = Image.fromarray(thresh)
    text2 = pytesseract.image_to_string(img2, lang='ara+eng', config='--psm 3')
    print("\nMethod 2 (Otsu):")
    print(text2)
    
    # Method 3: Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img3 = Image.fromarray(adaptive)
    text3 = pytesseract.image_to_string(img3, lang='ara+eng', config='--psm 3')
    print("\nMethod 3 (Adaptive):")
    print(text3)

if __name__ == "__main__":
    test_preprocessing()
