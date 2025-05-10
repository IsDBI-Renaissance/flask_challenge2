from typing import Dict, List, Any
from openai import OpenAI
import os

class ClassificationExplainer:
    def __init__(self, together_api_key: str = None):
        """
        Initialize the Classification Explainer
        
        Args:
            together_api_key: Together AI API key (optional, will use environment variable if not provided)
        """
        # Use provided API key or get from environment
        if together_api_key is None:
            together_api_key = os.environ.get("TOGETHER_API_KEY")
            
        if not together_api_key:
            raise ValueError("Together API key must be provided or set as TOGETHER_API_KEY environment variable")
            
        # Initialize Together AI client
        self.client = OpenAI(
            api_key=together_api_key,
            base_url="https://api.together.xyz/v1"
        )
    
    def explain_classification(self, journal_entries: List[Dict], context: str, standard: str) -> str:
        """
        Generate a detailed explanation for the classification decision
        
        Args:
            journal_entries: List of journal entry dictionaries
            context: Additional context for the transaction
            standard: The AAOIFI standard that was selected
            
        Returns:
            Detailed explanation string
        """
        # Format journal entries for the prompt
        entries_text = "\n".join([
            f"- {entry['account']}: Debit = {entry['debit']}, Credit = {entry['credit']}"
            for entry in journal_entries
        ])
        
        # Create the system prompt
        system_prompt = f"""
        You are an expert in Islamic finance and AAOIFI accounting standards. 
        Explain why the following journal entries were classified under {standard}.
        
        Journal entries:
        {entries_text}
        
        Additional context: {context}
        
        Your explanation should:
        1. Identify the key characteristics of the transaction that match the standard
        2. Explain the accounting treatment required by the standard
        3. Highlight any Shariah-compliance aspects
        4. Provide examples of similar transactions under this standard
        
        Structure your response with clear paragraphs and bullet points where appropriate.
        """
        
        # Query the Together AI API
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Explain this classification"}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        
        return response.choices[0].message.content