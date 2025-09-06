from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import os
import joblib
import pandas as pd
from openai import OpenAI

class BaseLoanService(ABC):
    """Base class for all loan services"""
    
    def __init__(self, model_path: str, openai_api_key: Optional[str] = None):
        self.model_path = model_path
        self.models = {}
        self.client = None
        
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        
        self.load_models()
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Return list of required fields for this loan type"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return system prompt for this loan type"""
        pass
    
    @abstractmethod
    def predict_loan(self, user_input: Dict[str, Any]) -> tuple:
        """Predict loan amount and interest rate"""
        pass
    
    def load_models(self):
        """Load ML models from the specified path"""
        try:
            model_files = self.get_model_files()
            for key, filename in model_files.items():
                full_path = os.path.join(self.model_path, filename)
                if os.path.exists(full_path):
                    self.models[key] = joblib.load(full_path)
                    print(f"Loaded {key} model from {full_path}")
                else:
                    print(f"Warning: Model file {full_path} not found")
                    self.models[key] = None
        except Exception as e:
            print(f"Error loading models: {e}")
    
    @abstractmethod
    def get_model_files(self) -> Dict[str, str]:
        """Return dictionary of model files needed"""
        pass
    
    def extract_info_from_response(self, user_text: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract information from user response using OpenAI or fallback logic"""
        # Try OpenAI with very short timeout first
        if self.client:
            extraction_prompt = self.get_extraction_prompt(user_text, conversation)
            
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.1,
                    max_tokens=500,
                    timeout=8  # 8 second timeout
                )
                extracted_text = resp.choices[0].message.content.strip()
                import re, json
                m = re.search(r"\{.*\}", extracted_text, re.DOTALL)
                if m:
                    return json.loads(m.group())
            except Exception as e:
                print(f"OpenAI extraction failed (using fallback): {e}")
        
        # Fallback to simple pattern matching for basic fields
        return self._fallback_extraction(user_text, conversation)
    
    def _fallback_extraction(self, user_text: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Fallback extraction using simple pattern matching.
        Enhanced to capture common loan fields when OpenAI is unavailable,
        so the bot doesn't keep asking the same question.
        """
        extracted: Dict[str, Any] = {}
        text_lower = user_text.lower().strip()

        import re

        # Helper to parse numbers like 5,00,000; 5 lakh; 5L; 1.2 cr
        def parse_amount(s: str) -> float | None:
            s = s.strip().lower()
            # Normalize Indian units
            try:
                if re.search(r"\b(cr|crore)s?\b", s):
                    num = re.sub(r"[^\d.]+", "", s)
                    return float(num) * 10000000
                if re.search(r"\b(l|lac|lakh)s?\b", s):
                    num = re.sub(r"[^\d.]+", "", s)
                    return float(num) * 100000
                # Remove commas and non-digit except dot
                num = re.sub(r"[^\d.]+", "", s.replace(",", ""))
                return float(num) if num else None
            except Exception:
                return None

        # Get last assistant message to infer which field was asked
        last_assistant_msg = ""
        for msg in reversed(conversation or []):
            if msg.get('role') == 'assistant':
                last_assistant_msg = (msg.get('content') or '').lower()
                break

        # 1) Name detection
        name_patterns = [
            r"my name is\s+([a-zA-Z\s]+)",
            r"i am\s+([a-zA-Z\s]+)",
            r"call me\s+([a-zA-Z\s]+)",
            r"i'm\s+([a-zA-Z\s]+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).strip().title()
                if len(name) > 1 and not any(word in name.lower() for word in ['years', 'old', 'work', 'job', 'salary']):
                    extracted['Customer_Name'] = name
                    break
        # If it looks like just a name after being asked for name
        if not extracted.get('Customer_Name') and 'name' in last_assistant_msg and len(text_lower.split()) <= 3:
            if re.match(r'^[a-zA-Z\s]+$', user_text.strip()) and 2 <= len(user_text.strip()) <= 50:
                extracted['Customer_Name'] = user_text.strip().title()

        # 2) Phone and Email
        phone_pattern = r'(\+?91[-\s]?)?([6-9]\d{9})'
        phone_match = re.search(phone_pattern, user_text)
        if phone_match:
            extracted['Customer_Phone'] = phone_match.group(2)

        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, user_text)
        if email_match:
            extracted['Customer_Email'] = email_match.group(0)

        # 3) Age
        age_patterns = [
            r'i am\s+(\d{1,2})\s+years?\s+old',
            r'my age is\s+(\d{1,2})',
            r'age\s*:?\s*(\d{1,2})',
            r'(\d{1,2})\s+years?\s+old'
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = int(match.group(1))
                if 18 <= age <= 80:
                    extracted['Age'] = age
                    break

        # 4) Education-loan specific common fields (driven by last question or keywords)
        # Academic_Score (0-100)
        if 'academic score' in last_assistant_msg or 'academic_score' in last_assistant_msg or 'score' in text_lower:
            m = re.search(r'(\d{1,3})(?:\s*/\s*100)?', user_text)
            if m:
                score = int(m.group(1))
                if 0 <= score <= 100:
                    extracted['Academic_Score'] = score

        # Intended_Course
        course_map = {
            'stem': 'STEM', 'mba': 'MBA', 'medicine': 'Medicine', 'finance': 'Finance',
            'law': 'Law', 'arts': 'Arts', 'other': 'Other'
        }
        for k, v in course_map.items():
            if re.search(rf"\b{k}\b", text_lower):
                extracted['Intended_Course'] = v
                break

        # University_Tier
        if re.search(r'\btier\s*1\b|\btier1\b|\bt1\b', text_lower):
            extracted['University_Tier'] = 'Tier1'
        elif re.search(r'\btier\s*2\b|\btier2\b|\bt2\b', text_lower):
            extracted['University_Tier'] = 'Tier2'
        elif re.search(r'\btier\s*3\b|\btier3\b|\bt3\b', text_lower):
            extracted['University_Tier'] = 'Tier3'

        # Coapplicant_Income
        if 'coapplicant' in text_lower or 'co-applicant' in text_lower or 'co applicant' in text_lower or 'coapplicant_income' in last_assistant_msg:
            amt = parse_amount(user_text)
            if amt is None:
                m = re.search(r'(?:income|salary)\D*(\d[\d,\.\s\w]*)', text_lower)
                if m:
                    amt = parse_amount(m.group(1))
            if amt is not None:
                extracted['Coapplicant_Income'] = amt

        # Guarantor_Networth
        if 'guarantor' in text_lower or 'networth' in text_lower or 'net worth' in text_lower:
            amt = parse_amount(user_text)
            if amt is None:
                m = re.search(r'(?:net\s*worth|worth)\D*(\d[\d,\.\s\w]*)', text_lower)
                if m:
                    amt = parse_amount(m.group(1))
            if amt is not None:
                extracted['Guarantor_Networth'] = amt

        # CIBIL_Score
        if 'cibil' in text_lower or 'cibil' in last_assistant_msg:
            m = re.search(r'(\d{3})', user_text)
            if m:
                extracted['CIBIL_Score'] = int(m.group(1))

        # Loan_Type (Secured / Unsecured)
        if 'secured' in text_lower:
            extracted['Loan_Type'] = 'Secured'
        elif 'unsecured' in text_lower:
            extracted['Loan_Type'] = 'Unsecured'

        # Loan_Term (years)
        if 'term' in last_assistant_msg or 'year' in text_lower or 'loan term' in last_assistant_msg:
            m = re.search(r'(\d{1,2})\s*(?:years?|yrs?)?', text_lower)
            if m:
                extracted['Loan_Term'] = int(m.group(1))

        # Expected_Loan_Amount
        if 'expected loan amount' in last_assistant_msg or 'loan amount' in text_lower or 'amount' in text_lower:
            amt = parse_amount(user_text)
            if amt is not None and amt > 0:
                extracted['Expected_Loan_Amount'] = amt

        return extracted
    
    @abstractmethod
    def get_extraction_prompt(self, user_text: str, conversation: List[Dict[str, str]]) -> str:
        """Get extraction prompt for this loan type"""
        pass
    
    def assistant_greeting(self, conversation: List[Dict[str, str]]) -> str:
        """Generate greeting message"""
        if not self.client:
            return self.get_fallback_greeting()
        
        try:
            # Create a proper greeting prompt
            greeting_messages = conversation.copy()
            greeting_messages.append({
                "role": "user", 
                "content": "Hello, I'm interested in this loan. Please greet me and ask for the first piece of information you need."
            })
            
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=greeting_messages,
                temperature=0.2,
                max_tokens=200,
                timeout=8
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"OpenAI greeting failed: {e}")
            return self.get_fallback_greeting()
    
    @abstractmethod
    def get_fallback_greeting(self) -> str:
        """Fallback greeting when OpenAI is not available"""
        pass
    
    def assistant_followup(self, conversation: List[Dict[str, str]], user_profile: Dict[str, Any], missing_fields: List[str]) -> str:
        """Generate follow-up message focused on ONLY the next missing field.
        This reduces loops/hallucinations by constraining the model output."""
        # If everything is collected, close politely
        if not missing_fields:
            return "Thank you for providing all the information!"
        
        next_field = missing_fields[0]
        
        # No OpenAI client → use deterministic, rule-based question
        if not self.client:
            return self.get_fallback_followup(missing_fields)
        
        # Strict system guidance for the model
        context_info = f"""
        You are completing a structured form step-by-step.
        Collected fields: {user_profile}
        Remaining fields (in order): {missing_fields}
        Your task: Ask ONE short question to collect ONLY the next field: {next_field}.
        Requirements:
        - Ask for {next_field} only, do not ask about other fields.
        - Provide a brief input format hint and one example value.
        - If the user's last message already included {next_field}, acknowledge it in a single short sentence and then ask for the next field from the remaining list.
        - Output just the question (and the one-sentence acknowledgement when applicable), nothing else.
        """
        conversation_copy = conversation.copy()
        conversation_copy.append({"role": "system", "content": context_info})
        
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_copy,
                temperature=0.1,  # lower temp for consistency
                max_tokens=180,
                timeout=8
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"OpenAI followup failed: {e}")
            return self.get_fallback_followup(missing_fields)
    
    def get_fallback_followup(self, missing_fields: List[str]) -> str:
        """Fallback follow-up: deterministic question per field to avoid repetition."""
        if not missing_fields:
            return "Thank you for providing all the information!"
        
        next_field = missing_fields[0]
        
        # Simple templates with format hints and examples
        templates = {
            "Customer_Name": "Please share your full name (e.g., Riya Sharma).",
            "Customer_Email": "What is your email address? (e.g., riya.sharma@example.com)",
            "Customer_Phone": "What is your 10-digit phone number? (digits only, e.g., 9876543210)",
            "Age": "What is your age in years? (e.g., 24)",
            "Academic_Score": "What's your academic score out of 100? (e.g., 82)",
            "Intended_Course": "Which course are you planning to pursue? Choose one: STEM, MBA, Medicine, Finance, Law, Arts, Other.",
            "University_Tier": "What is your university tier? Choose one: Tier1, Tier2, Tier3.",
            "Coapplicant_Income": "What is the annual co-applicant income in INR? (e.g., 600000)",
            "Guarantor_Networth": "What is the guarantor's total net worth in INR? (e.g., 1500000)",
            "CIBIL_Score": "What is your CIBIL score? (650–900, e.g., 720)",
            "Loan_Type": "Do you want a Secured loan (with collateral) or Unsecured loan (no collateral)?",
            "Loan_Term": "What loan term do you prefer in years? (1–15, e.g., 5)",
            "Expected_Loan_Amount": "What loan amount are you looking for in INR? (e.g., 800000)",
        }
        
        return templates.get(
            next_field,
            f"I'd like to know your {next_field.replace('_',' ').lower()}. Please provide it in a short, clear format."
        )