import re
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# External dependencies (assuming installed as per constraints)
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

class TextNormalizer:
    """Handles Arabic/English text normalization and digit conversion."""
    
    ARABIC_TO_ENGLISH_MAP = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')

    @staticmethod
    def normalize_digits(text: str) -> str:
        """Converts Arabic digits to Latin digits."""
        return text.translate(TextNormalizer.ARABIC_TO_ENGLISH_MAP)

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalizes whitespace and standardizes common symbols."""
        if not text:
            return ""
        
        # Normalize Arabic digits
        text = TextNormalizer.normalize_digits(text)
        
        # Remove common OCR artifacts (noise)
        # Specifically targeting some Arabic/Mixed character noise observed in screenshots
        text = re.sub(r'[|_~‎‏]+', '', text)
        
        # Standardize spaces and line breaks (keep line breaks but normalize them)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()

class DocumentOCR:
    """Handles extraction from PDFs (text-based or scanned) and images."""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd and pytesseract:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_from_pdf(self, pdf_path: str) -> str:
        """Extracts text using PyMuPDF, falling back to OCR if empty."""
        if not fitz:
            return "Error: PyMuPDF (fitz) not installed."
        
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            logging.error(f"Error reading PDF: {e}")
            return ""

        if len(text.strip()) < 10:  # Threshold for "scanned" check
            return self.extract_from_scanned_pdf(pdf_path)
        
        return text

    def extract_from_scanned_pdf(self, pdf_path: str) -> str:
        """Converts PDF pages to images and runs OCR."""
        if not pytesseract or not Image:
            return "Error: pytesseract or Pillow not installed."
        
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img, lang='ara+eng')
            doc.close()
        except Exception as e:
            logging.error(f"Error OCR-ing scanned PDF: {e}")
        return text

    def extract_from_image(self, image_path: str) -> str:
        """Directly runs OCR on an image file, trying multiple PSM modes."""
        if not pytesseract or not Image:
            return "Error: pytesseract or Pillow not installed."
        
        try:
            img = Image.open(image_path)
            # Try PSM 3 (auto) and 6 (uniform block) to get more text
            text3 = pytesseract.image_to_string(img, lang='ara+eng', config='--psm 3')
            text6 = pytesseract.image_to_string(img, lang='ara+eng', config='--psm 6')
            
            # Combine or pick best? Combining with separator is often better for rule-based
            return text3 + "\n---ALT-OCR---\n" + text6
        except Exception as e:
            logging.error(f"Error OCR-ing image: {e}")
            return ""

class TemplateManager:
    """Handles template loading and detection."""
    
    def __init__(self, templates_file: str):
        self.templates_file = templates_file
        self.templates = self.load_templates()

    def load_templates(self) -> List[Dict[str, Any]]:
        if not Path(self.templates_file).exists():
            return []
        try:
            with open(self.templates_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading templates: {e}")
            return []

    def detect_template(self, text: str) -> Optional[Dict[str, Any]]:
        """Scores templates based on keywords."""
        best_match = None
        max_score = 0
        
        text_lower = text.lower()
        
        for template in self.templates:
            score = 0
            keywords = template.get('keywords', [])
            for kw in keywords:
                if kw.lower() in text_lower:
                    score += 1
            
            if score > max_score:
                max_score = score
                best_match = template
                
        return best_match if max_score > 0 else None

class FinancialParser:
    """Extracts fields from text based on template rules."""
    
    def __init__(self, template: Optional[Dict[str, Any]], raw_text: str):
        self.template = template
        self.raw_text = raw_text
        self.raw_text_norm = TextNormalizer.clean_text(raw_text)
        self.results = {
            "template": template["name"] if template else "UNKNOWN",
            "confidence": "LOW",
            "amount": None,
            "currency": None,
            "transaction_id": None,
            "date": None,
            "sender": None,
            "receiver": None,
            "iban": None,
            "raw_text": raw_text,
            "evidence": {}
        }

    def parse(self) -> Dict[str, Any]:
        if not self.template:
            self._parse_generic()
        else:
            self._parse_with_template()
            
        # Specific fallback for currency if template is BANKAK and it's missing
        if self.template and self.template["name"] == "BANKAK" and not self.results.get("currency"):
            self.results["currency"] = "SDG"
            self.results["evidence"]["currency_fallback"] = "Defaulted to SDG for BANKAK"

        # Fallbacks for missing required fields
        if not self.results.get("amount"):
            self._greedy_amount_search()
        
        if not self.results.get("transaction_id"):
            self._greedy_transaction_id_search()

        if not self.results.get("sender"):
            self._greedy_sender_search()

        if not self.results.get("receiver"):
            self._greedy_receiver_search()
        
        # Always run greedy name search to find the best candidate (especially from ALT-OCR)
        self._greedy_receiver_name_search()
        
        if not self.results.get("date"):
            self._greedy_date_search()
        
        # Post-process to remove noise and handle RTL digit reversals
        self._post_process_results()
        
        self.results["confidence"] = self._calculate_confidence()
        return self.results

    def _post_process_results(self):
        """Final cleanup of results to remove noise."""
        EXCLUDE = ["التفاصيل", "تمت", "العملية", "نجاح", "معاملة", "Powered", "نفقات", "المزيد", "رقم الموبايل", "N/A", "التعليق", "إسم المرسل اليه", "To Account Title"]
        for field in ["sender", "receiver", "receiver_name"]:
            val = self.results.get(field)
            if val:
                # Remove common noise phrases from the middle of the text
                for ex in EXCLUDE:
                    val = val.replace(ex, "").strip()
                
                # Special handling for account numbers
                if field in ["sender", "receiver"]:
                    # Keep only digits for accounts
                    cleaned = re.sub(r'[^\d]', '', val)
                    
                    # Heuristic for RTL reversal: If it's Bankak and looks like a branch/account split
                    # Bankak accounts are usually 16-20 digits or groups of 4.
                    # e.g., 0160 1287 1717 0001 -> 1 1717 1287 0160
                    if self.template and self.template["name"] == "BANKAK":
                         # If it looks like it's missing trailing 0s and starts with 1
                         if len(cleaned) == 13 and cleaned.startswith("1"):
                              # OCR might have mangled it. Don't flip blindly but be aware.
                              pass
                    
                    val = cleaned
                
                # If the result is just noise or too short, clear it
                if len(val) < 3 or val.lower() == "n/a":
                    self.results[field] = None
                    if field in self.results["evidence"]:
                        del self.results["evidence"][field]
                else:
                    self.results[field] = val

    def _greedy_sender_search(self):
        """Looks for sender info by handle or by label."""
        # Clean phrases to ignore
        EXCLUDE = ["التفاصيل", "تمت", "العملية", "نجاح", "معاملة", "Powered", "نفقات"]

        # 1. Look for @ handles (flexible spacing and mangled domains)
        # We look for something before or after @
        handles = re.findall(r'([^\s@]+)\s*@\s*([^\s@]+)?', self.raw_text_norm)
        if handles:
            h = handles[0]
            handle_str = f"{h[0]}@{h[1] or 'instapay'}"
            if not any(ex in handle_str for ex in EXCLUDE):
                self.results["sender"] = handle_str
                self.results["evidence"]["sender_handle"] = f"{h[0]} @ {h[1]}"
                return

        # 2. Look for patterns like 'من' or 'From'
        match = re.search(r'(?:من|From)[:\s]*([^\n\r\|]{3,})', self.raw_text_norm)
        if match:
            val = match.group(1).strip()
            if not any(ex in val for ex in EXCLUDE):
                self.results["sender"] = self._clean_name(val)
                self.results["evidence"]["sender_greedy"] = match.group(0)

    def _greedy_transaction_id_search(self):
        """Looks for long numeric strings likely to be transaction IDs."""
        matches = re.findall(r'(\d{10,20})', self.raw_text_norm)
        if matches:
            # Prefer the one that isn't the IBAN or Account (if known)
            for m in matches:
                if len(m) >= 11:
                    # Avoid capturing the date as ID
                    if m not in (self.results.get("date") or ""):
                        self.results["transaction_id"] = m
                        self.results["evidence"]["id_greedy"] = m
                        break

    def _greedy_receiver_search(self):
        """Looks for receiver info by handle or by label."""
        EXCLUDE = ["التفاصيل", "تمت", "العملية", "نجاح", "معاملة", "Powered", "نفقات"]
        
        handles = re.findall(r'([^\s@]+)\s*@\s*([^\s@]+)?', self.raw_text_norm)
        if len(handles) > 1:
            h = handles[1]
            handle_str = f"{h[0]}@{h[1] or 'instapay'}"
            if not any(ex in handle_str for ex in EXCLUDE):
                self.results["receiver"] = handle_str
                self.results["evidence"]["receiver_handle"] = f"{h[0]} @ {h[1]}"
                return

        # 2. Look for patterns like 'إلى' or 'To'
        match = re.search(r'(?:إلى|To)[:\s]*([^\n\r\|]{3,})', self.raw_text_norm)
        if match:
            val = match.group(1).strip()
            if not any(ex in val for ex in EXCLUDE):
                self.results["receiver"] = self._clean_name(val)
                self.results["evidence"]["receiver_greedy"] = match.group(0)

    def _greedy_date_search(self):
        """Looks for date-like strings in the text."""
        # 1. DD-MMM-YYYY HH:MM:SS
        pattern1 = r'(\d{1,2}-[A-Za-z]{3,}-\d{4}\s+\d{2}:\d{2}:\d{2})'
        # 2. DD-MM-YYYY
        pattern2 = r'(\d{1,2}-\d{2}-\d{4})'
        # 3. YYYY-MM-DD
        pattern3 = r'(\d{4}-\d{2}-\d{1,2})'
        
        for p in [pattern1, pattern2, pattern3]:
            match = re.search(p, self.raw_text_norm)
            if match:
                self.results["date"] = match.group(1)
                self.results["evidence"]["date_greedy"] = match.group(0)
                return

    def _greedy_receiver_name_search(self):
        """Looks for receiver name near specific labels, prioritizing better matches across all OCR text."""
        labels = ["إسم المرسل اليه", "اسم المرسل اليه", "إسم المرسل إليه", "To Account Title"]
        candidates = []
        
        # Add existing result as a candidate if it exists
        if self.results.get("receiver_name"):
            existing = self.results["receiver_name"]
            arabic_chars = len(re.findall(r'[\u0600-\u06FF]', existing))
            candidates.append((len(existing) + (arabic_chars * 2), existing))

        for label in labels:
            matches = list(re.finditer(re.escape(label), self.raw_text_norm))
            for match_obj in matches:
                idx = match_obj.start()
                
                # ... (rest of search logic)
                line_start = self.raw_text_norm.rfind('\n', 0, idx) + 1
                line_end = self.raw_text_norm.find('\n', idx)
                if line_end == -1: line_end = len(self.raw_text_norm)
                
                current_line = self.raw_text_norm[line_start:line_end].replace(label, '').strip(': \t')
                
                prev_line = ""
                all_prev = self.raw_text_norm[:idx].strip().split('\n')
                if all_prev:
                    prev_line = all_prev[-1].strip()

                for cand in [current_line, prev_line]:
                    if len(cand) > 3 and not any(ex in cand for ex in ["من", "إلى", "الى", "حساب", "رقم", "المبلغ", "العملية"]):
                        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', cand))
                        score = len(cand) + (arabic_chars * 2)
                        candidates.append((score, cand))

        if candidates:
            # Pick the best scoring candidate
            best_cand = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
            self.results["receiver_name"] = self._clean_name(best_cand)
            if best_cand not in (self.results.get("receiver_name") or ""):
                 self.results["evidence"]["receiver_name_greedy_improved"] = best_cand

    def _clean_name(self, name: str) -> str:
        """Removes common OCR noise from names."""
        # Remove any leading/trailing punctuation or noise characters
        name = re.sub(r'^[‎‏\s\n*._-]+|[‎‏\s\n*._-]+$', '', name)
        # Remove excessive stars (masking)
        name = re.sub(r'\*+', '*', name)
        return name

    def _greedy_amount_search(self):
        """Looks for numbers near currency symbols if regex failed."""
        # Clean the text even more for greedy search (flatten it)
        flat_text = self.raw_text_norm.replace('\n', ' ')
        
        patterns = [
            r'([\d,]{2,}(?:\.\d{2})?)\s*(?:EGP|جنيه|SDG|SAR)',
            r'(?:EGP|جنيه|SDG|SAR)\s*([\d,]{2,}(?:\.\d{2})?)',
            r'(?:مبلغ|Amount|مجع|معع|مج)[:\s]*([\d,]{2,}(?:\.\d{2})?)'
        ]
        for p in patterns:
            match = re.search(p, flat_text, re.IGNORECASE)
            if match:
                val = self._clean_amount(match.group(1))
                if val:
                    self.results["amount"] = val
                    self.results["evidence"]["amount_greedy"] = match.group(0)
                    break

    def _parse_with_template(self):
        fields = self.template.get("fields", {})
        for field, config in fields.items():
            pattern = config.get("pattern")
            if not pattern:
                continue
            
            match = re.search(pattern, self.raw_text_norm, re.IGNORECASE | re.MULTILINE)
            if match:
                group_idx = config.get("group", 1)
                match_val = match.group(group_idx)
                if match_val:
                    value = match_val.strip()
                    
                    if field == "amount":
                        value = self._clean_amount(value)
                    
                    self.results[field] = value
                    self.results["evidence"][field] = match.group(0).strip()

    def _parse_generic(self):
        """Fallback generic parsing logic."""
        amount_match = re.search(r'(?:Total|Amount|مبلغ)[:\s]*([\d,]+\.?\d*)', self.raw_text_norm, re.IGNORECASE)
        if amount_match:
            self.results["amount"] = self._clean_amount(amount_match.group(1))
            self.results["evidence"]["amount"] = amount_match.group(0)

        id_match = re.search(r'(?:Ref|ID|رقم العملية)[:\s]*(\w+)', self.raw_text_norm, re.IGNORECASE)
        if id_match:
            self.results["transaction_id"] = id_match.group(1)
            self.results["evidence"]["transaction_id"] = id_match.group(0)

    def _clean_amount(self, value: str) -> Optional[float]:
        try:
            # Remove all non-numeric/non-dot/non-comma characters
            clean_val = re.sub(r'[^\d.,]', '', value)
            
            # Handle mixed separators (e.g., 1.250,50 or 1,250.50)
            if ',' in clean_val and '.' in clean_val:
                # Assume the last one is the decimal
                if clean_val.find(',') > clean_val.find('.'):
                    clean_val = clean_val.replace('.', '').replace(',', '.')
                else:
                    clean_val = clean_val.replace(',', '')
            else:
                # Only one type of separator or none
                # If it's a comma and there are 2 digits after it, it's likely decimal
                # But in Egypt, usually comma is thousands. Let's assume comma is thousands for now.
                clean_val = clean_val.replace(',', '')
                
            return float(clean_val)
        except:
            return None

    def _calculate_confidence(self) -> str:
        # High confidence if template matched and at least amount + something else found
        required_fields = ["amount", "transaction_id", "receiver"]
        found_count = sum(1 for f in required_fields if self.results.get(f))
        
        if self.results["template"] != "UNKNOWN" and found_count >= 2:
            return "HIGH"
        elif found_count >= 1:
            return "MEDIUM"
        else:
            return "LOW"

def process_document(file_path: str, templates_path: str, tesseract_cmd: Optional[str] = None, template_name: Optional[str] = None):
    """Main pipeline for a single document."""
    ocr_engine = DocumentOCR(tesseract_cmd=tesseract_cmd)
    path = Path(file_path)
    
    if path.suffix.lower() == '.pdf':
        raw_text = ocr_engine.extract_from_pdf(str(path))
    else:
        raw_text = ocr_engine.extract_from_image(str(path))
    
    tm = TemplateManager(templates_path)
    
    # Manual override or auto-detect
    template = None
    if template_name and template_name != "Auto-Detect":
        for t in tm.templates:
            if t["name"] == template_name:
                template = t
                break
    
    if not template:
        template = tm.detect_template(raw_text)
    
    parser = FinancialParser(template, raw_text)
    return parser.parse()

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Financial Document OCR Parser")
    parser.add_argument("file", help="Path to PDF or Image file")
    parser.add_argument("--templates", default="templates.json", help="Path to templates JSON")
    
    args = parser.parse_args()
    
    # Configure Tesseract path from user input
    tesseract_path = r"D:\tesseract\tesseract.exe"
    ocr_engine = DocumentOCR(tesseract_cmd=tesseract_path)
    
    if not Path(args.file).exists():
        print(f"Error: File {args.file} not found.")
        sys.exit(1)
        
    try:
        result = process_document(args.file, args.templates)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error processing: {e}")
