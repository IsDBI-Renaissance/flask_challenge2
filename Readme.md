AAOIFI Reverse Transaction Classifier API
========================================

A REST API that classifies accounting journal entries into AAOIFI (Islamic Finance) standards using rule-based patterns and AI-powered semantic analysis.

Problem It Solves
-----------------
- Manual classification of Islamic finance transactions is time-consuming and error-prone
- Lack of standardization in mapping journal entries to AAOIFI standards
- Audit/compliance challenges in verifying Islamic accounting compliance

Key Features:
- Rule-based pattern matching for common transaction types
- GPT-4 powered semantic analysis for nuanced classification
- Explainable AI results with reasoning for each classification
- Support for all major AAOIFI standards (FAS 1, 4, 7, 10, 20, 28, 32)

File Structure
-------------
.
├── app.py                # Flask API endpoints
├── classifier.py         # Core classification logic
├── explainer.py          # AI-powered explanation generator
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md             # Documentation

Setup & Installation
--------------------
Prerequisites:
- Python 3.8+
- OpenAI API key

1. Clone repository:
   git clone https://github.com/your-repo/aaoifi-classifier.git
   cd aaoifi-classifier

2. Install dependencies:
   pip install -r requirements.txt

3. Configure environment:
   Create .env file with:
   OPENAI_API_KEY=your_openai_key_here

4. Run the API:
   Development: python app.py
   Production: gunicorn --bind 0.0.0.0:5000 app:app

API Endpoints
-------------

1. Health Check
GET /health
Response:
{"status": "OK", "message": "Service is running"}

2. Classify Transaction
POST /api/v1/classify
Request Body:
{
  "journal_entries": [
    {"account": "Murabaha Receivable", "debit": 100000, "credit": 0},
    {"account": "Murabaha Asset", "debit": 0, "credit": 80000}
  ],
  "additional_context": "Optional description"
}

Response:
{
  "most_likely_standard": "FAS_28",
  "standard_probabilities": [
    {
      "standard": "FAS_28",
      "probability": 0.92,
      "reason": "Matches Murabaha pattern"
    }
  ],
  "key_features": ["Murabaha accounts detected"],
  "detailed_explanation": "AI-generated reasoning..."
}

3. Analyze Pattern
POST /api/v1/analyze-pattern
Request Body:
{
  "account_pattern": ["Right of Use Asset", "Ijarah Liability"],
  "standard_id": "FAS_32"
}

Response:
{
  "standard_id": "FAS_32",
  "standard_name": "Ijarah standard",
  "account_analysis": [...],
  "overall_match_score": 0.85,
  "common_journal_entries": [...]
}

Supported AAOIFI Standards
-------------------------
- FAS 1: General Presentation
- FAS 4: Foreign Currency
- FAS 7: Investments
- FAS 10: Istisna'a
- FAS 20: Musharakah
- FAS 28: Murabaha
- FAS 32: Ijarah

Error Handling
-------------
400 Bad Request - Invalid input format
500 Server Error - Internal processing failure

Example Error Response:
{
  "error": "Missing journal_entries",
  "message": "The journal_entries field is required"
}

Example Use Cases
----------------
1. Classifying Murabaha transactions
2. Verifying Ijarah contract entries
3. Auditing foreign currency transactions

License
-------
MIT License
