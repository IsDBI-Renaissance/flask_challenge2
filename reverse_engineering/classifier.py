import os
import json
import re
from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
from openai import OpenAI

class ReverseTransactionClassifier:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the Reverse Transaction Classifier
        
        Args:
            openai_api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        # Use provided API key or get from environment
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
            
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
        # Load standards knowledge base
        self.standards_knowledge = self._load_standards_knowledge()
        
        # Define standard patterns
        self.standard_patterns = self._define_standard_patterns()
        
        # Define standard feature extractors
        self.feature_extractors = {
            "account_names": self._extract_account_names_feature,
            "debit_credit_ratio": self._extract_debit_credit_ratio,
            "transaction_structure": self._extract_transaction_structure
        }
        
    def _load_standards_knowledge(self) -> Dict:
        """
        Load knowledge base on AAOIFI standards
        
        Returns:
            Dict containing structured knowledge on standards
        """
        standards = {
            "FAS_1": {
                "name": "General Presentation and Disclosure in the Financial Statements",
                "key_terms": ["financial statements", "presentation", "disclosure", 
                             "balance sheet", "income statement"],
                "typical_accounts": ["Assets", "Liabilities", "Equity", "Revenue", "Expenses"],
                "common_patterns": ["Increasing disclosure accounts", "Financial statement structuring"]
            },
            "FAS_4": {
                "name": "Foreign Currency Transactions and Foreign Operations",
                "key_terms": ["foreign currency", "exchange rate", "translation", "monetary items", 
                             "non-monetary items", "foreign operation"],
                "typical_accounts": ["Foreign Currency", "Exchange Difference", "Translation Reserve", 
                                    "Foreign Operations", "Monetary Assets", "Non-Monetary Assets", "Equity"],
                "common_patterns": ["Currency translation", "Foreign operation consolidation", 
                                   "Exchange rate adjustments", "Equity acquisition"]
            },
            "FAS_7": {
                "name": "Investments",
                "key_terms": ["investment", "equity method", "fair value", "acquisition", 
                             "associates", "subsidiaries", "joint ventures"],
                "typical_accounts": ["Investment", "Investment in Associates", "Investment in Subsidiaries", 
                                    "Fair Value Reserve", "Equity", "Cash"],
                "common_patterns": ["Investment acquisition", "Investment disposal", 
                                   "Fair value adjustment", "Dividend recording"]
            },
            "FAS_10": {
                "name": "Istisna'a and Parallel Istisna'a",
                "key_terms": ["istisna'a", "manufacturer", "contract", "work-in-progress", 
                             "percentage of completion", "parallel istisna'a", "cost plus"],
                "typical_accounts": ["Istisna'a Receivables", "Istisna'a Work-in-Progress", 
                                    "Parallel Istisna'a Payable", "Accounts Payable", "Cash", 
                                    "Istisna'a Revenue", "Cost of Istisna'a"],
                "common_patterns": ["Contract recognition", "Work-in-progress recording", 
                                   "Revenue recognition", "Cost reversal"]
            },
            "FAS_20": {
                "name": "Musharakah and Constant Musharakah",
                "key_terms": ["musharakah", "partnership", "joint venture", "equity participation",
                             "profit sharing", "capital contribution"],
                "typical_accounts": ["Musharakah Investment", "Cash", "Musharakah Capital", 
                                    "Musharakah Profits", "Partners' Equity", "Expenses"],
                "common_patterns": ["Capital contribution", "Profit distribution", 
                                   "Loss allocation", "Partnership dissolution"]
            },
            "FAS_28": {
                "name": "Murabaha and Other Deferred Payment Sales",
                "key_terms": ["murabaha", "cost-plus sale", "deferred payment", "profit", 
                             "installment", "commodity"],
                "typical_accounts": ["Murabaha Asset", "Murabaha Receivable", "Cash", 
                                    "Deferred Profit", "Inventory", "Cost of Sales"],
                "common_patterns": ["Asset acquisition", "Murabaha sale", 
                                   "Profit recognition", "Installment collection"]
            },
            "FAS_32": {
                "name": "Ijarah and Ijarah Muntahia Bittamleek",
                "key_terms": ["ijarah", "lease", "rental", "muntahia bittamleek", "right of use", 
                             "usufruct", "transfer of ownership"],
                "typical_accounts": ["Right of Use Asset", "Ijarah Asset", "Deferred Ijarah Cost", 
                                    "Ijarah Liability", "Cash", "Rental Revenue", "Depreciation"],
                "common_patterns": ["Initial recognition", "Periodic rental", 
                                   "Asset depreciation", "Ownership transfer"]
            }
        }
        return standards
        
    def _define_standard_patterns(self) -> Dict:
        """
        Define common journal entry patterns for each standard
        
        Returns:
            Dict mapping patterns to standards
        """
        patterns = {
            # FAS 4 patterns
            "foreign_currency_transaction": {
                "accounts": ["Asset", "Cash"],
                "narrative": "Foreign currency purchase of assets",
                "standard": "FAS_4",
                "confidence": 0.7
            },
            "equity_acquisition": {
                "accounts": ["Equity", "Cash"],
                "narrative": "Equity acquisition potentially related to foreign operations",
                "standard": "FAS_4",
                "confidence": 0.7
            },
            
            # FAS 7 patterns
            "investment_acquisition": {
                "accounts": ["Investment", "Cash"],
                "narrative": "Acquisition of investment instruments",
                "standard": "FAS_7",
                "confidence": 0.8
            },
            "investment_fair_value": {
                "accounts": ["Investment", "Fair Value Reserve"],
                "narrative": "Fair value adjustment of investments",
                "standard": "FAS_7",
                "confidence": 0.85
            },
            
            # FAS 10 patterns
            "istisna_contract": {
                "accounts": ["Istisna'a Receivables", "Istisna'a Revenue"],
                "narrative": "Recognition of Istisna'a contract",
                "standard": "FAS_10",
                "confidence": 0.9
            },
            "work_in_progress": {
                "accounts": ["Work-in-Progress", "Accounts Payable"],
                "narrative": "Recording work in progress for Istisna'a",
                "standard": "FAS_10",
                "confidence": 0.75
            },
            "cost_reversal": {
                "accounts": ["Accounts Payable", "Work-in-Progress"],
                "narrative": "Cost reversal for Istisna'a contract",
                "standard": "FAS_10",
                "confidence": 0.8
            },
            
            # FAS 28 patterns
            "murabaha_acquisition": {
                "accounts": ["Murabaha Asset", "Cash"],
                "narrative": "Acquisition of asset for Murabaha",
                "standard": "FAS_28",
                "confidence": 0.85
            },
            "murabaha_sale": {
                "accounts": ["Murabaha Receivable", "Murabaha Asset", "Deferred Profit"],
                "narrative": "Sale of asset under Murabaha contract",
                "standard": "FAS_28",
                "confidence": 0.9
            },
            
            # FAS 32 patterns
            "ijarah_initial": {
                "accounts": ["Right of Use Asset", "Deferred Ijarah Cost", "Ijarah Liability"],
                "narrative": "Initial recognition of Ijarah contract",
                "standard": "FAS_32",
                "confidence": 0.9
            },
            "ijarah_rental": {
                "accounts": ["Cash", "Ijarah Revenue", "Deferred Ijarah Cost"],
                "narrative": "Recording of Ijarah rental payment",
                "standard": "FAS_32",
                "confidence": 0.85
            }
        }
        return patterns
    
    def _extract_account_names_feature(self, journal_entries: List[Dict]) -> Dict:
        """
        Extract features related to account names
        
        Args:
            journal_entries: List of journal entry dictionaries
            
        Returns:
            Dict containing account name features
        """
        accounts = [entry["account"].lower() for entry in journal_entries]
        
        # Check for standard-specific keywords in account names
        standard_scores = {}
        
        for std_id, std_info in self.standards_knowledge.items():
            score = 0
            hits = []
            
            # Check typical accounts
            for account in accounts:
                for typical_account in std_info["typical_accounts"]:
                    if typical_account.lower() in account:
                        score += 1
                        hits.append(f"Account '{account}' matches typical account '{typical_account}'")
            
            # Check key terms in account names
            for account in accounts:
                for term in std_info["key_terms"]:
                    if term.lower() in account:
                        score += 0.5
                        hits.append(f"Account '{account}' contains key term '{term}'")
            
            # Normalize score based on number of entries
            normalized_score = score / max(len(journal_entries), 1)
            
            standard_scores[std_id] = {
                "score": normalized_score,
                "evidence": hits
            }
        
        return {
            "feature": "account_names",
            "standard_scores": standard_scores
        }
    
    def _extract_debit_credit_ratio(self, journal_entries: List[Dict]) -> Dict:
        """
        Extract features related to debit/credit structure
        
        Args:
            journal_entries: List of journal entry dictionaries
            
        Returns:
            Dict containing debit/credit ratio features
        """
        total_debits = sum(entry["debit"] for entry in journal_entries)
        total_credits = sum(entry["credit"] for entry in journal_entries)
        
        # Number of debit and credit entries
        num_debits = sum(1 for entry in journal_entries if entry["debit"] > 0)
        num_credits = sum(1 for entry in journal_entries if entry["credit"] > 0)
        
        # Check balanced nature of transaction
        is_balanced = abs(total_debits - total_credits) < 0.01
        
        # Calculate complexity score (more entries = more complex)
        complexity = len(journal_entries) / 2  # Normalize
        
        return {
            "feature": "debit_credit_ratio",
            "total_debits": total_debits,
            "total_credits": total_credits,
            "num_debit_entries": num_debits,
            "num_credit_entries": num_credits,
            "is_balanced": is_balanced,
            "complexity": complexity
        }
    
    def _extract_transaction_structure(self, journal_entries: List[Dict]) -> Dict:
        """
        Extract features related to transaction structure
        
        Args:
            journal_entries: List of journal entry dictionaries
            
        Returns:
            Dict containing transaction structure features
        """
        # Extract account structure patterns
        debit_accounts = [entry["account"] for entry in journal_entries if entry["debit"] > 0]
        credit_accounts = [entry["account"] for entry in journal_entries if entry["credit"] > 0]
        
        # Match against known patterns
        pattern_matches = []
        pattern_scores = {}
        
        for pattern_name, pattern_info in self.standard_patterns.items():
            pattern_accounts = set(pattern_info["accounts"])
            entry_accounts = set([entry["account"].split()[0] for entry in journal_entries])
            
            # Calculate overlap between pattern accounts and transaction accounts
            overlap = pattern_accounts.intersection(entry_accounts)
            match_ratio = len(overlap) / max(len(pattern_accounts), 1)
            
            if match_ratio > 0.5:  # If more than half the accounts match
                standard_id = pattern_info["standard"]
                confidence = pattern_info["confidence"] * match_ratio
                
                pattern_matches.append({
                    "pattern_name": pattern_name,
                    "match_ratio": match_ratio,
                    "standard": standard_id,
                    "narrative": pattern_info["narrative"],
                    "confidence": confidence
                })
                
                # Update pattern scores for each standard
                if standard_id not in pattern_scores:
                    pattern_scores[standard_id] = {
                        "score": 0,
                        "matches": []
                    }
                
                pattern_scores[standard_id]["score"] += confidence
                pattern_scores[standard_id]["matches"].append({
                    "pattern": pattern_name,
                    "confidence": confidence
                })
        
        return {
            "feature": "transaction_structure",
            "pattern_matches": pattern_matches,
            "pattern_scores": pattern_scores
        }
    
    def _extract_features(self, journal_entries: List[Dict]) -> List[Dict]:
        """
        Extract all features from journal entries
        
        Args:
            journal_entries: List of journal entry dictionaries
            
        Returns:
            List of feature dictionaries
        """
        features = []
        
        for feature_name, extractor in self.feature_extractors.items():
            feature = extractor(journal_entries)
            features.append(feature)
        
        return features
    
    def _semantic_analysis(self, journal_entries: List[Dict], additional_context: str = "") -> Dict:
        """
        Perform semantic analysis using LLM
        
        Args:
            journal_entries: List of journal entry dictionaries
            additional_context: Additional context for the transaction
            
        Returns:
            Dict containing semantic analysis results
        """
        # Format journal entries for the prompt
        entries_text = "\n".join([
            f"- {entry['account']}: Debit = {entry['debit']}, Credit = {entry['credit']}"
            for entry in journal_entries
        ])
        
        # Create a summary of all standards for the prompt
        standards_summary = "\n".join([
            f"- {std_id}: {details['name']} (Keywords: {', '.join(details['key_terms'][:3])}...)"
            for std_id, details in self.standards_knowledge.items()
        ])
        
        # Create the system prompt
        system_prompt = f"""
        You are an expert in Islamic finance and AAOIFI accounting standards. 
        Analyze the journal entries below and determine which AAOIFI standard they most likely relate to.
        
        Focus on these AAOIFI standards:
        {standards_summary}
        
        In your analysis, consider:
        1. The account names and what they suggest about the transaction type
        2. The debit/credit structure and what it implies
        3. Any contextual information provided
        4. Typical patterns for different Islamic finance contracts
        
        Journal entries to analyze:
        {entries_text}
        
        Additional context: {additional_context}
        
        Provide:
        1. The most likely standard (use format FAS_X)
        2. Probability score for top 3 potential standards (0-1)
        3. Reasoning for each potential match
        4. Key features of the transaction that influenced your decision
        
        Format your response as valid JSON with keys:
        "most_likely_standard", "probabilities" (array of objects with "standard", "score", "reasoning" keys), and "key_features" (array of strings).
        """
        
        # Query the LLM
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt}
            ],
            temperature=0.3,  # Low temperature for more deterministic results
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except (json.JSONDecodeError, KeyError):
            # Fallback to default results if parsing fails
            return {
                "most_likely_standard": "FAS_4",  # Default to common standard
                "probabilities": [
                    {"standard": "FAS_4", "score": 0.5, "reasoning": "Default fallback due to parsing error"}
                ],
                "key_features": ["Unable to extract features from LLM response"]
            }
    
    def _aggregate_results(self, features: List[Dict], semantic_analysis: Dict) -> Dict:
        """
        Aggregate results from feature extraction and semantic analysis
        
        Args:
            features: List of extracted features
            semantic_analysis: Results from semantic analysis
            
        Returns:
            Dict containing aggregated results
        """
        # Initialize scores for each standard
        standard_scores = {std_id: 0.0 for std_id in self.standards_knowledge.keys()}
        standard_evidence = {std_id: [] for std_id in self.standards_knowledge.keys()}
        
        # Process account names feature
        for feature in features:
            if feature["feature"] == "account_names":
                for std_id, score_info in feature["standard_scores"].items():
                    standard_scores[std_id] += score_info["score"] * 0.3  # 30% weight
                    # Add evidence if score is significant
                    if score_info["score"] > 0.1 and score_info["evidence"]:
                        standard_evidence[std_id].extend(score_info["evidence"][:2])  # Top 2 pieces of evidence
        
        # Process transaction structure feature
        for feature in features:
            if feature["feature"] == "transaction_structure" and "pattern_scores" in feature:
                for std_id, pattern_info in feature["pattern_scores"].items():
                    standard_scores[std_id] += pattern_info["score"] * 0.4  # 40% weight
                    # Add matching patterns as evidence
                    for match in pattern_info["matches"]:
                        standard_evidence[std_id].append(
                            f"Matches pattern '{match['pattern']}' with {match['confidence']:.2f} confidence"
                        )
        
        # Add semantic analysis results
        for prob in semantic_analysis.get("probabilities", []):
            std_id = prob["standard"]
            score = prob["score"]
            
            if std_id in standard_scores:
                standard_scores[std_id] += score * 0.3  # 30% weight
                standard_evidence[std_id].append(f"Semantic analysis: {prob['reasoning']}")
        
        # Normalize scores
        total_score = sum(standard_scores.values())
        if total_score > 0:
            for std_id in standard_scores:
                standard_scores[std_id] /= total_score
        
        # Sort standards by score
        sorted_standards = sorted(
            [(std_id, score) for std_id, score in standard_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top 3 standards
        top_standards = [
            {
                "standard": std_id,
                "probability": score,
                "reason": "\n".join(standard_evidence[std_id][:3])  # Top 3 reasons
            }
            for std_id, score in sorted_standards[:3] if score > 0.05  # Only standards with significant scores
        ]
        
        # Get most likely standard
        most_likely_standard = sorted_standards[0][0] if sorted_standards else "Unknown"
        
        # Extract key features from semantic analysis
        key_features = semantic_analysis.get("key_features", [])
        
        # Add debit/credit structure features
        for feature in features:
            if feature["feature"] == "debit_credit_ratio":
                key_features.append(f"Transaction has {feature['num_debit_entries']} debit entries and {feature['num_credit_entries']} credit entries")
                if feature["is_balanced"]:
                    key_features.append("Transaction is balanced (total debits = total credits)")
        
        return {
            "most_likely_standard": most_likely_standard,
            "standard_probabilities": top_standards,
            "key_features": key_features
        }
    
    def classify_transaction(self, journal_entries: List[Dict], additional_context: str = "") -> Dict:
        """
        Classify journal entries to identify applicable AAOIFI standard
        
        Args:
            journal_entries: List of journal entry dictionaries
            additional_context: Additional context for the transaction
            
        Returns:
            Dict containing classification results
        """
        # Extract features from journal entries
        features = self._extract_features(journal_entries)
        
        # Perform semantic analysis
        semantic_analysis = self._semantic_analysis(journal_entries, additional_context)
        
        # Aggregate results
        results = self._aggregate_results(features, semantic_analysis)
        
        return results
    
    def analyze_pattern(self, account_pattern: List[str], standard_id: str) -> Dict:
        """
        Analyze how strongly a specific account pattern matches a standard
        
        Args:
            account_pattern: List of account names
            standard_id: AAOIFI standard ID to analyze against
            
        Returns:
            Dict containing pattern analysis
        """
        if standard_id not in self.standards_knowledge:
            return {"error": f"Unknown standard: {standard_id}"}
        
        standard_info = self.standards_knowledge[standard_id]
        
        # Check account matches
        account_matches = []
        for account in account_pattern:
            matches = []
            for typical_account in standard_info["typical_accounts"]:
                if typical_account.lower() in account.lower() or account.lower() in typical_account.lower():
                    matches.append(typical_account)
            
            account_matches.append({
                "account": account,
                "matches": matches,
                "match_score": len(matches) / max(len(standard_info["typical_accounts"]), 1)
            })
        
        # Check for similar patterns in standard patterns
        pattern_matches = []
        for pattern_name, pattern_info in self.standard_patterns.items():
            if pattern_info["standard"] == standard_id:
                # Calculate overlap
                pattern_set = set(pattern_info["accounts"])
                input_set = set(account_pattern)
                
                overlap = pattern_set.intersection(input_set)
                match_score = len(overlap) / max(len(pattern_set), 1)
                
                if match_score > 0.3:  # If there's some meaningful overlap
                    pattern_matches.append({
                        "pattern_name": pattern_name,
                        "pattern_accounts": pattern_info["accounts"],
                        "narrative": pattern_info["narrative"],
                        "match_score": match_score
                    })
        
        # Overall match score
        overall_score = sum(m["match_score"] for m in account_matches) / len(account_matches) if account_matches else 0
        
        # Get common journal entries for this standard
        common_entries = self._get_common_entries(standard_id)
        
        return {
            "standard_id": standard_id,
            "standard_name": standard_info["name"],
            "account_analysis": account_matches,
            "similar_patterns": pattern_matches,
            "overall_match_score": overall_score,
            "common_journal_entries": common_entries
        }
    
    def _get_common_entries(self, standard_id: str) -> List[Dict]:
        """
        Get common journal entries for a standard
        
        Args:
            standard_id: AAOIFI standard ID
            
        Returns:
            List of common journal entry templates
        """
        # This would ideally be expanded with more comprehensive examples
        common_entries = {
            "FAS_4": [
                {
                    "description": "Recording a foreign currency purchase",
                    "entries": [
                        {"account": "Asset", "debit": "FC amount * exchange rate", "credit": 0},
                        {"account": "Cash/Bank", "debit": 0, "credit": "FC amount * exchange rate"}
                    ]
                }
            ],
            "FAS_7": [
                {
                    "description": "Recording an investment acquisition",
                    "entries": [
                        {"account": "Investment", "debit": "Purchase amount", "credit": 0},
                        {"account": "Cash/Bank", "debit": 0, "credit": "Purchase amount"}
                    ]
                }
            ],
            "FAS_10": [
                {
                    "description": "Recording an Istisna'a contract",
                    "entries": [
                        {"account": "Istisna'a Receivable", "debit": "Contract value", "credit": 0},
                        {"account": "Istisna'a Revenue", "debit": 0, "credit": "Contract value"}
                    ]
                }
            ],
            "FAS_28": [
                {
                    "description": "Recording a Murabaha sale",
                    "entries": [
                        {"account": "Murabaha Receivable", "debit": "Selling price", "credit": 0},
                        {"account": "Murabaha Asset", "debit": 0, "credit": "Cost price"},
                        {"account": "Deferred Profit", "debit": 0, "credit": "Selling price - Cost price"}
                    ]
                }
            ],
            "FAS_32": [
                {
                    "description": "Initial recognition of Ijarah",
                    "entries": [
                        {"account": "Right of Use Asset", "debit": "Asset value", "credit": 0},
                        {"account": "Deferred Ijarah Cost", "debit": "Total rentals - Asset value", "credit": 0},
                        {"account": "Ijarah Liability", "debit": 0, "credit": "Total rentals"}
                    ]
                }
            ]
        }
        
        return common_entries.get(standard_id, [])