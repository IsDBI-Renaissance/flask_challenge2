from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from reverse_engineering.classifier import ReverseTransactionClassifier
from reverse_engineering.explainer import ClassificationExplainer

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize the classifier
classifier = ReverseTransactionClassifier(
    together_api_key=os.getenv("TOGETHER_API_KEY")  # Changed parameter name
)

# Initialize the explainer
explainer = ClassificationExplainer(
    together_api_key=os.getenv("TOGETHER_API_KEY")  # Changed parameter name
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "OK", "message": "Service is running"}), 200

@app.route('/api/v1/classify', methods=['POST'])
def classify_transaction():
    """
    Classify journal entries to identify the applicable AAOIFI standard
    
    Request JSON format:
    {
        "journal_entries": [
            {"account": "GreenTech Equity", "debit": 1750000, "credit": 0},
            {"account": "Cash", "debit": 0, "credit": 1750000}
        ],
        "additional_context": "Optional additional context"
    }
    """
    try:
        data = request.json
        
        if not data or "journal_entries" not in data:
            return jsonify({
                "error": "Invalid request. 'journal_entries' is required."
            }), 400
            
        journal_entries = data["journal_entries"]
        additional_context = data.get("additional_context", "")
        
        # Get classification results
        classification_results = classifier.classify_transaction(
            journal_entries, 
            additional_context
        )
        
        # Get detailed explanation
        explanation = explainer.explain_classification(
            journal_entries,
            additional_context,
            classification_results["most_likely_standard"]
        )
        
        # Combine results
        result = {
            **classification_results,
            "detailed_explanation": explanation
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/api/v1/analyze-pattern', methods=['POST'])
def analyze_pattern():
    """
    Analyze a specific journal entry pattern
    
    Request JSON format:
    {
        "account_pattern": ["Asset", "Cash"],
        "standard_id": "FAS_32"
    }
    """
    try:
        data = request.json
        
        if not data or "account_pattern" not in data or "standard_id" not in data:
            return jsonify({
                "error": "Invalid request. 'account_pattern' and 'standard_id' are required."
            }), 400
            
        account_pattern = data["account_pattern"]
        standard_id = data["standard_id"]
        
        pattern_analysis = classifier.analyze_pattern(account_pattern, standard_id)
        
        return jsonify(pattern_analysis), 200
        
    except Exception as e:
        app.logger.error(f"Error analyzing pattern: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)