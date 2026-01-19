import json
from financial_ocr import FinancialParser

# Mock raw text from an InstaPay screenshot
mock_instapay_text = """
InstaPay
Success
Amount: 1,250.50 EGP
Transaction ID: ABC123456789
Date: 12-01-2025
To: Ahmed Ali
Bank: NBE
"""

# Mock templates
templates = [
    {
        "name": "INSTAPAY",
        "keywords": ["InstaPay", "Success"],
        "fields": {
            "amount": { "pattern": "Amount:\\s*([\\d,]+\\.?\\d*)", "group": 1 },
            "currency": { "pattern": "(EGP|USD)", "group": 1 },
            "transaction_id": { "pattern": "Transaction ID:\\s*(\\w+)", "group": 1 },
            "receiver": { "pattern": "To:\\s*([^\\n]+)", "group": 1 }
        }
    }
]

# Run mock parsing
parser = FinancialParser(templates[0], mock_instapay_text)
result = parser.parse()

print("Parsed Mock Data:")
print(json.dumps(result, indent=2))
