
import json
import os
from pathlib import Path
import sys

# Ensure we can import from the current directory
sys.path.append(str(Path(r"d:\work\newocr")))

from financial_ocr import FinancialParser, TemplateManager, TextNormalizer

def test_user_snippets():
    # Combining the user's snippets
    raw_text = """
--- PSM 3 ---
(em) تفاصيل المعاملة

1003 0935 7585 0001

O2FHZS/O2FHU4

©" مشاركة ١ ##طباعة | O تحميل
©2024 بنك | لخرطوم |بنكك حساب

--- PSM 4 ---
رقم العملية 20028380982
التاريخ والوقت 18:40:19 18-Jan-2026
نوع العملية تحويل إلى حساب آخر

©" مشاركة ١ ##طباعة | O تحميل
©2024 بنك | لخرطوم |بنكك حساب

--- PSM 6 ---
ak
تفاصيل المعاملة
تحميل O | ##طباعة ١ مشاركة "©
بنك | لخرطوم |بنكك حساب 2024©

--- PSM 11 ---
sak

رجوع>»

© مشاركة

©2024 بنك | لخرطوم |بنكك حساب
    """
    # Note: The user's snippet DOES NOT contain the amount '2240000.00' or the name.
    # I should check if my greedy search can find the account and ID though.
    # And maybe I'll manually add the amount line to see if it works when present.
    
    tm = TemplateManager(r"d:\work\newocr\templates.json")
    template = None
    for t in tm.templates:
        if t["name"] == "BANKAK_TRANSACTION_DETAILS":
            template = t
            break
            
    parser = FinancialParser(template, raw_text)
    result = parser.parse()
    
    print("\n--- RESULTS ---")
    print(f"Template: {result['template']}")
    print(f"Amount: {result['amount']} {result['currency']}")
    print(f"Trx ID: {result['transaction_id']}")
    print(f"Date: {result['date']}")
    print(f"Sender: {result['sender']}")
    print(f"Receiver: {result['receiver']}")
    print(f"Receiver Name: {result['receiver_name']}")
    
    # Check evidence
    print("\n--- EVIDENCE ---")
    for k, v in result['evidence'].items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    test_user_snippets()
