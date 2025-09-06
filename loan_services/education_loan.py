from typing import Dict, List, Any, Tuple
import pandas as pd
import re
import json
from .base_loan import BaseLoanService

class EducationLoanService(BaseLoanService):
    """Education Loan Service"""
    
    def get_required_fields(self) -> List[str]:
        return [
            "Customer_Name",
            "Customer_Email", 
            "Customer_Phone",
            "Age",
            "Academic_Score",  # Changed from Academic_Performance to Academic_Score
            "Intended_Course",
            "University_Tier",
            "Coapplicant_Income",
            "Guarantor_Networth",
            "CIBIL_Score",
            "Loan_Type",
            "Loan_Term",
            "Expected_Loan_Amount",
        ]
    
    def get_model_files(self) -> Dict[str, str]:
        return {
            "xgb_loan": "xgb_loan_amount_v2.pkl",
            "xgb_interest": "xgb_interest_rate_v2.pkl", 
            "encoders": "encoders_v2.pkl",
            "scaler": "scaler_v2.pkl",
        }
    
    def get_system_prompt(self) -> str:
        return """You are a friendly and professional education loan advisor chatbot.

Your task is to systematically collect the following information from users through natural conversation:

Required Fields (collect in this order):
1. Customer_Name (full name)
2. Customer_Email (email address)
3. Customer_Phone (phone number - MUST be exactly 10 digits)
4. Age (MUST be between 18-35 for education loan applicants)
5. Academic_Score (ask for score out of 100, NOT performance level)
6. Intended_Course: one of ["STEM","MBA","Medicine","Finance","Law","Arts","Other"]
7. University_Tier: one of ["Tier1","Tier2","Tier3"]
8. Coapplicant_Income (annual income in INR - MUST be positive)
9. Guarantor_Networth (total assets value in INR - MUST be positive)
10. CIBIL_Score (MUST be between 650-900 for eligibility)
11. Loan_Type: "Secured" (requires collateral) or "Unsecured" (no collateral needed)
12. Loan_Term (MUST be between 1-15 years for education loans)
13. Expected_Loan_Amount (MUST be positive and not exceed ₹3,00,00,000)

VALIDATION RULES - STRICTLY ENFORCE:
- Phone: Exactly 10 digits, reject if invalid
- Age: 18-35 only, reject if outside range
- Academic Score: Ask for number 0-100, convert to grade (90-100=Excellent, 75-89=Good, 60-74=Average, <60=Poor)
- CIBIL: 650-900 required, reject if below 650 (not eligible)
- Loan Amount: Max ₹3,00,00,000, reject if exceeded (not eligible)
- Loan Term: 1-15 years only, reject if outside range
- All monetary values: Must be positive (no negative numbers like -56418)
- Always explain secured vs unsecured: "Secured loans require collateral, unsecured loans don't"

Guidelines:
1) ALWAYS start by asking for their name, email, and phone number first
2) Be conversational and friendly, not robotic
3) Ask 1-2 related questions at a time, don't overwhelm
4) VALIDATE each response according to rules above
5) If invalid, explain why and ask again
6) For academic performance, ask "What's your academic score out of 100?"
7) When you have ALL valid information, say exactly: INFORMATION_COMPLETE
8) Do NOT provide loan advice or predictions - only collect information

Start by introducing yourself and asking for their name first."""
    
    def get_fallback_greeting(self) -> str:
        return "Hello! I'm here to help you with your education loan application. To get started, may I have your full name please?"
    
    def extract_info_from_response(self, user_text: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract information from user response with enhanced fallback logic"""
        print(f"DEBUG - Starting extraction for: '{user_text}'")
        
        # Try OpenAI first if available
        if self.client:
            extraction_prompt = self.get_extraction_prompt(user_text, conversation)
            
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0
                )
                extracted_text = resp.choices[0].message.content.strip()
                m = re.search(r"\{.*\}", extracted_text, re.DOTALL)
                if m:
                    extracted = json.loads(m.group())
                    print(f"DEBUG - OpenAI extracted: {extracted}")
                    if extracted:  # Only return if we got something useful
                        return extracted
            except Exception as e:
                print(f"DEBUG - OpenAI extraction failed: {e}")
        
        # Enhanced fallback extraction
        return self._enhanced_fallback_extraction(user_text, conversation)
    
    def _enhanced_fallback_extraction(self, user_text: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Enhanced fallback extraction with context awareness"""
        extracted = {}
        text_lower = user_text.lower().strip()
        
        # Get context from last assistant message
        last_assistant_msg = ""
        if len(conversation) > 0:
            for msg in reversed(conversation):
                if msg.get('role') == 'assistant':
                    last_assistant_msg = msg.get('content', '').lower()
                    break
        
        # 1. CUSTOMER NAME
        name_patterns = [
            r"(?:my name is|i am|i'm|call me|name:|this is)\s+([a-zA-Z\s]{2,30})",
            r"^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*$",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                name = match.group(1).strip().title()
                if not any(word in name.lower() for word in ['years', 'old', 'score', 'percent', 'tier']):
                    extracted['Customer_Name'] = name
                    break
        
        # Context-aware name extraction
        if not extracted.get('Customer_Name') and ('name' in last_assistant_msg or 'call you' in last_assistant_msg):
            words = user_text.strip().split()
            if 1 <= len(words) <= 3 and all(re.match(r'^[a-zA-Z]+$', word) for word in words):
                if not any(word.lower() in ['yes', 'no', 'ok', 'sure', 'hello', 'hi'] for word in words):
                    extracted['Customer_Name'] = user_text.strip().title()
        
        # 2. PHONE NUMBER
        phone_patterns = [
            r'(?:\+?91[\s-]?)?([6-9]\d{9})',
            r'(?:phone|mobile|contact|number)[\s:]*(\+?91)?[\s-]*([6-9]\d{9})',
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, user_text)
            if match:
                phone = match.group(-1)
                if len(phone) == 10 and phone[0] in '6789':
                    extracted['Customer_Phone'] = phone
                    break
        
        # Context-aware phone extraction
        if not extracted.get('Customer_Phone') and any(word in last_assistant_msg for word in ['phone', 'mobile', 'contact', 'number']):
            phone_digits = re.sub(r'[^\d]', '', user_text)
            if len(phone_digits) == 10 and phone_digits[0] in '6789':
                extracted['Customer_Phone'] = phone_digits
        
        # 3. EMAIL
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, user_text)
        if email_match:
            extracted['Customer_Email'] = email_match.group(0)
        
        # 4. AGE
        age_patterns = [
            r'(?:age|years?\s*old|yrs?)\s*(?:is\s*)?(?::|=)?\s*(\d{1,2})',
            r'(\d{1,2})\s*(?:years?\s*old|yrs?)',
            r'i am\s*(\d{1,2})'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = int(match.group(1))
                if 18 <= age <= 35:  # Education loan specific age range
                    extracted['Age'] = age
                    break
        
        # Context-aware age extraction
        if not extracted.get('Age') and 'age' in last_assistant_msg:
            age_match = re.search(r'\b(\d{1,2})\b', user_text)
            if age_match:
                age = int(age_match.group(1))
                if 18 <= age <= 35:
                    extracted['Age'] = age
        
        # 5. EDUCATION LOAN SPECIFIC FIELDS
        
        # Academic Score extraction
        academic_patterns = [
            r'(?:academic\s*score|score|percentage|percent|marks?)\s*(?:is\s*)?(?::|=)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent|percentage)?',
            r'(?:got|scored|have|my)\s*(\d+(?:\.\d+)?)\s*(?:%|percent|percentage|marks?|score)',
        ]
        
        for pattern in academic_patterns:
            match = re.search(pattern, text_lower)
            if match:
                score = float(match.group(1))
                if score > 100:
                    score = min(score / 10, 100)  # Handle 850/10 = 85%
                if 0 <= score <= 100:
                    extracted["Academic_Score"] = score
                    break
        
        # Intended Course extraction
        course_mapping = {
            'engineering': 'STEM', 'computer science': 'STEM', 'cs': 'STEM', 'tech': 'STEM',
            'mba': 'MBA', 'management': 'MBA', 'business': 'MBA',
            'medicine': 'Medicine', 'medical': 'Medicine', 'mbbs': 'Medicine', 'doctor': 'Medicine',
            'finance': 'Finance', 'banking': 'Finance', 'accounting': 'Finance',
            'law': 'Law', 'legal': 'Law', 'llb': 'Law',
            'arts': 'Arts', 'humanities': 'Arts', 'design': 'Arts',
            'other': 'Other', 'different': 'Other'
        }
        
        for keyword, course in course_mapping.items():
            if keyword in text_lower:
                extracted["Intended_Course"] = course
                break
        
        # University Tier extraction
        tier_patterns = {
            'iit|iim|bits|nit|top|premier|tier1|tier 1': 'Tier1',
            'good|decent|average|tier2|tier 2': 'Tier2',
            'local|small|private|tier3|tier 3': 'Tier3'
        }
        
        for pattern, tier in tier_patterns.items():
            if re.search(pattern, text_lower):
                extracted["University_Tier"] = tier
                break
        
        # Coapplicant Income extraction
        income_patterns = [
            r'(?:coapplicant|co-applicant|parent|family).*?income.*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'income.*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:parent|family)',
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount > 0:
                    extracted['Coapplicant_Income'] = amount
                    break
        
        # Guarantor Networth extraction
        networth_patterns = [
            r'(?:guarantor|assets|networth|property).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:assets|property|worth)',
        ]
        
        for pattern in networth_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount > 0:
                    extracted['Guarantor_Networth'] = amount
                    break
        
        # CIBIL Score extraction
        cibil_patterns = [
            r'(?:cibil|credit.*?score)\s*(?:is\s*)?(?::|=)?\s*(\d{3})',
            r'(\d{3})\s*(?:cibil|credit.*?score)',
        ]
        
        for pattern in cibil_patterns:
            match = re.search(pattern, text_lower)
            if match:
                score = int(match.group(1))
                if 300 <= score <= 900:
                    extracted['CIBIL_Score'] = score
                    break
        
        # Loan Type extraction
        if 'secured' in text_lower or 'collateral' in text_lower:
            extracted['Loan_Type'] = 'Secured'
        elif 'unsecured' in text_lower or 'no collateral' in text_lower:
            extracted['Loan_Type'] = 'Unsecured'
        
        # Loan Term extraction
        term_patterns = [
            r'(?:term|duration|years?)\s*(?:is\s*)?(?::|=)?\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?).*?(?:loan.*?term|duration)',
        ]
        
        for pattern in term_patterns:
            match = re.search(pattern, text_lower)
            if match:
                term = int(match.group(1))
                if 1 <= term <= 15:
                    extracted['Loan_Term'] = term
                    break
        
        # Expected Loan Amount extraction
        loan_amount_patterns = [
            r'(?:loan.*?amount|need.*?loan|want.*?loan).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:loan.*?amount|need.*?loan)',
        ]
        
        for pattern in loan_amount_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount > 0:
                    extracted['Expected_Loan_Amount'] = amount
                    break
        
        print(f"DEBUG - Fallback extracted: {extracted}")
        return extracted
    
    def convert_amount_to_number(self, amount_str: str) -> float:
        """Convert lakh/crore amounts to numbers with error handling"""
        if isinstance(amount_str, (int, float)):
            return float(amount_str)
            
        amount_str = str(amount_str).lower().strip()
        
        # Remove currency symbols and extra spaces
        amount_str = re.sub(r'[₹rs\.\s]+', '', amount_str)
        
        # Extract number and unit
        number_match = re.search(r'([\d,]+(?:\.[\d,]+)?)', amount_str)
        if not number_match:
            return 0.0
        
        # Remove commas and convert to float
        number_str = number_match.group(1).replace(',', '')
        try:
            number = float(number_str)
        except ValueError:
            return 0.0
        
        if 'crore' in amount_str:
            return number * 10000000  # 1 crore = 1,00,00,000
        elif 'lakh' in amount_str:
            return number * 100000    # 1 lakh = 1,00,000
        else:
            return number
    
    def validate_field(self, field_name: str, value: Any) -> Tuple[bool, str]:
        """Validate individual field values according to education loan rules"""
        try:
            if field_name == "Customer_Name":
                if not value or not str(value).strip():
                    return False, "Please provide your full name."
                name = str(value).strip()
                if len(name) < 2:
                    return False, "Please provide your complete name."
                return True, ""
                
            elif field_name == "Customer_Email":
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, str(value)):
                    return False, "Please provide a valid email address."
                return True, ""
                
            elif field_name == "Customer_Phone":
                phone_str = str(value).replace(' ', '').replace('-', '').replace('(', '').replace(')', '').replace('+91', '')
                if not phone_str.isdigit() or len(phone_str) != 10 or phone_str[0] not in '6789':
                    return False, "Invalid phone number. Phone number must be exactly 10 digits starting with 6, 7, 8, or 9."
                return True, ""
                
            elif field_name == "Age":
                age = float(value)
                if age < 18 or age > 35:
                    return False, "Invalid age. For education loan applicants, age must be between 18-35."
                return True, ""
                
            elif field_name == "Academic_Score":
                score = float(value)
                if score < 0:
                    return False, "Invalid score. Please enter a real quantity (score cannot be negative)."
                elif score > 100:
                    return False, "Invalid score. Please enter a real quantity (score cannot exceed 100)."
                return True, ""
                
            elif field_name == "CIBIL_Score":
                cibil = int(value)
                if cibil < 650:
                    return False, "You are not eligible. CIBIL score must be at least 650 for education loan."
                elif cibil > 900:
                    return False, "Invalid CIBIL score. CIBIL score cannot exceed 900."
                return True, ""
                
            elif field_name == "Expected_Loan_Amount":
                amount = float(value)
                if amount <= 0:
                    return False, "Invalid loan amount. All values must be positive."
                elif amount > 30000000:  # 3 crores
                    return False, "Not eligible. Loan amount cannot exceed ₹3,00,00,000."
                return True, ""
                
            elif field_name == "Loan_Term":
                term = int(value)
                if term <= 0:
                    return False, "Invalid loan term. All values must be positive."
                elif term < 1 or term > 15:
                    return False, "Invalid loan term. Education loan term must be between 1-15 years."
                return True, ""
                
            elif field_name in ["Coapplicant_Income", "Guarantor_Networth"]:
                amount = float(value)
                if amount <= 0:
                    return False, f"Invalid {field_name.lower().replace('_', ' ')}. All values must be positive (negative values like -56418 are not possible)."
                return True, ""
                
            elif field_name == "Intended_Course":
                valid_courses = ["STEM", "MBA", "Medicine", "Finance", "Law", "Arts", "Other"]
                if value not in valid_courses:
                    return False, f"Please select your intended course from: {', '.join(valid_courses)}"
                return True, ""
                
            elif field_name == "University_Tier":
                valid_tiers = ["Tier1", "Tier2", "Tier3"]
                if value not in valid_tiers:
                    return False, f"Please select university tier from: {', '.join(valid_tiers)}"
                return True, ""
                
            elif field_name == "Loan_Type":
                valid_types = ["Secured", "Unsecured"]
                if value not in valid_types:
                    return False, f"Please select loan type from: {', '.join(valid_types)}"
                return True, ""
                
            return True, ""
            
        except (ValueError, TypeError):
            field_display = field_name.replace('_', ' ').lower()
            return False, f"Please provide a valid {field_display} in the correct format."

    def convert_academic_score_to_performance(self, score: float) -> str:
        """Convert numeric academic score to performance grade"""
        if 90 <= score <= 100:
            return "Excellent"
        elif 75 <= score < 90:
            return "Good"
        elif 60 <= score < 75:
            return "Average"
        else:
            return "Poor"

    def get_extraction_prompt(self, user_text: str, conversation: List[Dict[str, str]]) -> str:
        return f"""
Based on the conversation history and the user's latest response, extract any education loan-related information.

Conversation so far: {conversation[-3:] if len(conversation) > 3 else conversation}

User's latest response: "{user_text}"

Extract information for these fields (only if mentioned):
- Customer_Name: full name as text
- Customer_Email: email address as text
- Customer_Phone: phone number as text (must be 10 digits)
- Age: number (must be 18-35)
- Academic_Score: number (0-100, will be converted to performance grade)
- Intended_Course: exactly one of ["STEM","MBA","Medicine","Finance","Law","Arts","Other"]
- University_Tier: exactly one of ["Tier1","Tier2","Tier3"]
- Coapplicant_Income: number in INR (must be positive)
- Guarantor_Networth: number in INR (must be positive)
- CIBIL_Score: number (must be 650-900)
- Loan_Type: exactly one of ["Secured","Unsecured"]
- Loan_Term: number (must be 1-15 years)
- Expected_Loan_Amount: number in INR (must be positive, max 30000000)

Return ONLY a JSON object with the extracted fields. If no information is found, return empty JSON {{}}.
Example: {{"Customer_Name": "John Doe", "Customer_Email": "john@example.com", "Age": 25, "Academic_Score": 85}}
""".strip()
    
    def repayment_capacity(self, income: float, networth: float, cibil: float) -> float:
        """Calculate repayment capacity for education loans"""
        return (income * 4) + (networth * 0.05) + (cibil / 2)
    
    def predict_loan(self, user_input: Dict[str, Any]) -> tuple:
        """Predict education loan amount and interest rate"""
        if not all([self.models.get("xgb_loan"), self.models.get("xgb_interest"), 
                   self.models.get("encoders"), self.models.get("scaler")]):
            raise ValueError("Required models not loaded")
        
        # Create a copy to avoid modifying original
        processed_input = user_input.copy()
        
        # Convert academic score to performance grade
        processed_input["Academic_Performance"] = self.convert_academic_score_to_performance(
            processed_input["Academic_Score"]
        )
        
        # Add computed feature
        processed_input["Repayment_Capacity"] = self.repayment_capacity(
            processed_input["Coapplicant_Income"],
            processed_input["Guarantor_Networth"],
            processed_input["CIBIL_Score"]
        )
        
        # Encode categorical variables
        encoders = self.models["encoders"]
        for col in ["Academic_Performance", "Intended_Course", "University_Tier", "Loan_Type"]:
            if col not in encoders:
                raise ValueError(f"Encoder for {col} not found.")
            processed_input[col] = encoders[col].transform([processed_input[col]])[0]
        
        # Prepare features for prediction
        features = [
            "Age", "Academic_Performance", "Intended_Course", "University_Tier",
            "Coapplicant_Income", "Guarantor_Networth", "CIBIL_Score",
            "Loan_Type", "Repayment_Capacity", "Loan_Term"
        ]
        
        X = pd.DataFrame([{k: processed_input[k] for k in features}])
        
        # Scale numeric features
        numeric_cols = ["Age", "Coapplicant_Income", "Guarantor_Networth", 
                       "CIBIL_Score", "Repayment_Capacity", "Loan_Term"]
        X[numeric_cols] = self.models["scaler"].transform(X[numeric_cols])
        
        # Make predictions
        loan_amt = self.models["xgb_loan"].predict(X)[0]
        interest = self.models["xgb_interest"].predict(X)[0]
        
        return round(float(loan_amt)), round(float(interest), 2)