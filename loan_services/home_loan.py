from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import re
import json
from .base_loan import BaseLoanService

class HomeLoanService(BaseLoanService):
    """Home Loan Service with XGBoost Model Integration"""
    
    def get_required_fields(self) -> List[str]:
        return [
            "Customer_Name",
            "Customer_Email",
            "Customer_Phone",
            "Age",
            "Income",  # Changed from Monthly_Income to match model
            "Guarantor_income",
            "Tenure",  # Changed from Loan_Term to match model
            "CIBIL_score",  # Changed from CIBIL_Score to match model
            "Employment_type",  # Changed from Employment_Type to match model
            "Down_payment",
            "Existing_total_EMI",  # Changed from Existing_EMI to match model
            "Loan_amount_requested",  # Changed from Expected_Loan_Amount to match model
            "Property_value",  # Changed from Property_Value to match model
        ]
    
    def get_model_files(self) -> Dict[str, str]:
        return {
            "loan_amount_model": "loan_amount_model.pkl",
            "interest_rate_model": "interest_rate_model.pkl",
        }
    
    def get_system_prompt(self) -> str:
        return """You are a friendly and professional home loan advisor chatbot.

Your task is to systematically collect the following information from users through natural conversation:

Required Customer Information:
- Customer_Name (full name)
- Customer_Email (email address)
- Customer_Phone (10-digit phone number)

Required Loan Information:
- Age (21-50 years - strict eligibility criteria)
- Income (monthly income in INR)
- Guarantor_income (guarantor's monthly income in INR, can be 0 if no guarantor)
- Tenure (loan repayment period in years, typically 5-30)
- CIBIL_score (credit score minimum 650 required for eligibility)
- Employment_type: exactly one of ["Business Owner", "Salaried", "Government Employee", "Self-Employed"]
- Down_payment (amount you can pay upfront in INR)
- Existing_total_EMI (current monthly EMI obligations in INR, can be 0)
- Loan_amount_requested (desired loan amount in INR)
- Property_value (total property value in INR)

Guidelines:
1) Be conversational and friendly, like a helpful bank representative.
2) Ask 1-2 related questions at a time to avoid overwhelming the customer.
3) Provide brief explanations when needed (e.g., "CIBIL score is your credit score", "LTV is loan-to-value ratio").
4) If user provides partial info, acknowledge it positively and ask for missing details.
5) Validate responses and ask for clarification if unclear.
6) For Employment_type, ensure it's exactly one of the four options.
7) When you have ALL information, say exactly: INFORMATION_COMPLETE
8) Do NOT provide loan predictions - only collect information professionally.

Start by introducing yourself as a home loan specialist and asking about their home buying plans."""
    
    def get_fallback_greeting(self) -> str:
        return "Hello! I'm a home loan specialist. I'm here to help you with your home loan application. Let's start with your full name - what should I call you?"
    
    def get_extraction_prompt(self, user_text: str, conversation: List[Dict[str, str]]) -> str:
        return f"""
Based on the conversation history and the user's latest response, extract any home loan-related information.

Conversation so far: {conversation[-3:] if len(conversation) > 3 else conversation}

User's latest response: "{user_text}"

Extract information for these fields (only if clearly mentioned):

Customer Information:
- Customer_Name: full name as string
- Customer_Email: email address as string  
- Customer_Phone: 10-digit phone number as string (remove +91, spaces, dashes)

Loan Information:
- Age: number (21-50, strict requirement)
- Income: number in INR (monthly income, must be positive)
- Guarantor_income: number in INR (guarantor's monthly income, 0 if none)
- Tenure: number (loan term in years, 5-30)
- CIBIL_score: number (minimum 650 required)
- Employment_type: exactly one of ["Business Owner", "Salaried", "Government Employee", "Self-Employed"]
- Down_payment: number in INR (upfront payment amount)
- Existing_total_EMI: number in INR (current monthly EMIs, 0 if none)
- Loan_amount_requested: number in INR (desired loan amount)
- Property_value: number in INR (total property value)

Important:
- For Employment_type, map variations like "business", "govt", "self employed" to exact options
- Convert lakhs/crores to actual numbers (e.g., "50 lakhs" = 5000000)
- Extract only information that is clearly stated

Return ONLY a JSON object with the extracted fields. If no information is found, return empty JSON {{}}.
Example: {{"Customer_Name": "John Doe", "Age": 35, "Employment_type": "Salaried", "Income": 80000, "Property_value": 5000000}}
""".strip()
    
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
                if not any(word in name.lower() for word in ['years', 'old', 'work', 'job', 'score']):
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
                if 21 <= age <= 50:  # Home loan specific age range
                    extracted['Age'] = age
                    break
        
        # Context-aware age extraction
        if not extracted.get('Age') and 'age' in last_assistant_msg:
            age_match = re.search(r'\b(\d{1,2})\b', user_text)
            if age_match:
                age = int(age_match.group(1))
                if 21 <= age <= 50:
                    extracted['Age'] = age
        
        # 5. HOME LOAN SPECIFIC FIELDS
        
        # Monthly Income extraction
        income_patterns = [
            r'(?:monthly.*?income|salary|earn.*?monthly).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:monthly.*?income|salary)',
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount > 0:
                    extracted['Income'] = amount
                    break
        
        # Guarantor Income extraction
        guarantor_patterns = [
            r'(?:guarantor.*?income|guarantor.*?salary).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:guarantor.*?income)',
        ]
        
        for pattern in guarantor_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount >= 0:
                    extracted['Guarantor_income'] = amount
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
                    extracted['CIBIL_score'] = score
                    break
        
        # Employment Type extraction
        employment_mapping = {
            'business': 'Business Owner', 'business owner': 'Business Owner', 'entrepreneur': 'Business Owner',
            'salaried': 'Salaried', 'employee': 'Salaried', 'job': 'Salaried',
            'government': 'Government Employee', 'govt': 'Government Employee', 'public sector': 'Government Employee',
            'self employed': 'Self-Employed', 'self-employed': 'Self-Employed', 'freelance': 'Self-Employed', 'consultant': 'Self-Employed'
        }
        
        for keyword, employment in employment_mapping.items():
            if keyword in text_lower:
                extracted['Employment_type'] = employment
                break
        
        # Down Payment extraction
        down_payment_patterns = [
            r'(?:down.*?payment|advance).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:down.*?payment|advance)',
        ]
        
        for pattern in down_payment_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount >= 0:
                    extracted['Down_payment'] = amount
                    break
        
        # Existing EMI extraction
        emi_patterns = [
            r'(?:existing.*?emi|current.*?emi).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:emi.*?payment)',
        ]
        
        for pattern in emi_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount >= 0:
                    extracted['Existing_total_EMI'] = amount
                    break
        
        # Loan Amount extraction
        loan_amount_patterns = [
            r'(?:loan.*?amount|need.*?loan|want.*?loan).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:loan.*?amount|need.*?loan)',
        ]
        
        for pattern in loan_amount_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount > 0:
                    extracted['Loan_amount_requested'] = amount
                    break
        
        # Property Value extraction
        property_patterns = [
            r'(?:property.*?value|house.*?value|home.*?price).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:property.*?value|house.*?price)',
        ]
        
        for pattern in property_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount > 0:
                    extracted['Property_value'] = amount
                    break
        
        # Tenure extraction
        tenure_patterns = [
            r'(?:tenure|duration|years?)\s*(?:is\s*)?(?::|=)?\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?).*?(?:tenure|duration)',
        ]
        
        for pattern in tenure_patterns:
            match = re.search(pattern, text_lower)
            if match:
                tenure = int(match.group(1))
                if 5 <= tenure <= 30:  # Home loan specific tenure range
                    extracted['Tenure'] = tenure
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
        """Validate individual field values with user-friendly messages"""
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
                    return False, "Please provide a valid 10-digit mobile number starting with 6, 7, 8, or 9."
                return True, ""
                
            elif field_name == "Age":
                age = float(value)
                if not (21 <= age <= 50):
                    return False, "I need your age to be between 21 and 50 years for home loan eligibility. Could you please confirm your age?"
                return True, ""
                
            elif field_name == "CIBIL_score":
                cibil = float(value)
                if cibil < 650:
                    return False, "Sorry, for home loans we require a minimum CIBIL score of 650. Unfortunately, your current score doesn't meet our eligibility criteria."
                elif not (300 <= cibil <= 900):
                    return False, "Your CIBIL score should be between 300 and 900. Could you please check and provide your correct credit score?"
                return True, ""
                
            elif field_name == "Employment_type":
                valid_types = ["Business Owner", "Salaried", "Government Employee", "Self-Employed"]
                if value not in valid_types:
                    return False, f"For employment type, please choose from: {', '.join(valid_types)}. Which category best describes your employment?"
                return True, ""
                
            elif field_name == "Tenure":
                tenure = float(value)
                if not (5 <= tenure <= 30):
                    return False, "Loan tenure should be between 5 and 30 years. How many years would you like to repay the loan?"
                return True, ""
                
            elif field_name == "Income":
                amount = float(value)
                if amount <= 0:
                    return False, "Could you please tell me your monthly income? This helps me calculate your loan eligibility."
                return True, ""
                
            elif field_name == "Property_value":
                amount = float(value)
                if amount <= 0:
                    return False, "What's the total value of the property you're planning to purchase? This is important for calculating your loan amount."
                return True, ""
                
            elif field_name == "Loan_amount_requested":
                amount = float(value)
                if amount <= 0:
                    return False, "How much loan amount are you looking for? Please share your expected loan requirement."
                return True, ""
                
            elif field_name == "Down_payment":
                amount = float(value)
                if amount < 0:
                    return False, "How much can you pay as down payment? Even if it's zero, please let me know."
                return True, ""
                
            elif field_name == "Guarantor_income":
                amount = float(value)
                if amount < 0:
                    return False, "Guarantor income cannot be negative. Please provide the guarantor's monthly income (enter 0 if no guarantor)."
                return True, ""
                
            elif field_name == "Existing_total_EMI":
                amount = float(value)
                if amount < 0:
                    return False, "Existing EMI cannot be negative. Please provide your current monthly EMI obligations (enter 0 if none)."
                return True, ""
                
            return True, ""
            
        except (ValueError, TypeError):
            field_display = field_name.replace('_', ' ').lower()
            return False, f"I didn't quite understand the {field_display}. Could you please provide it in a clear format?"
    
    def validate_complete_data(self, user_input: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate complete data including cross-field validations"""
        try:
            # Check if loan amount is greater than property value
            if 'Loan_amount_requested' in user_input and 'Property_value' in user_input:
                loan_amount = float(user_input['Loan_amount_requested'])
                property_value = float(user_input['Property_value'])
                
                if loan_amount > property_value:
                    return False, f"The loan amount requested (₹{loan_amount:,.0f}) cannot be more than the property value (₹{property_value:,.0f}). Please adjust your loan amount or property value."
            
            return True, ""
            
        except (ValueError, TypeError) as e:
            return False, "There was an error validating your information. Please check all the values you provided."
    
    def prepare_model_input(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data for the XGBoost model with feature engineering"""
        try:
            # Validate complete data first
            is_valid, error_msg = self.validate_complete_data(user_input)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Create input dataframe
            input_data = {
                'Age': float(user_input['Age']),
                'Income': float(user_input['Income']),
                'Guarantor_income': float(user_input.get('Guarantor_income', 0)),
                'Tenure': float(user_input['Tenure']),
                'CIBIL_score': float(user_input['CIBIL_score']),
                'Down_payment': float(user_input['Down_payment']),
                'Existing_total_EMI': float(user_input.get('Existing_total_EMI', 0)),
                'Loan_amount_requested': float(user_input['Loan_amount_requested']),
                'Property_value': float(user_input['Property_value']),
                'Employment_type': user_input['Employment_type']
            }
            
            print(f"Home Loan Input data prepared: {input_data}")
            
            input_df = pd.DataFrame([input_data])
            
            # Ensure Guarantor_income is numeric
            input_df['Guarantor_income'] = pd.to_numeric(input_df['Guarantor_income'], errors='coerce').fillna(0)
            
            # Feature Engineering: Calculate ratios (matching training code)
            input_df['LTV'] = input_df['Loan_amount_requested'] / input_df['Property_value']
            input_df['EMI_to_income'] = input_df['Existing_total_EMI'] / input_df['Income']
            input_df['DP_ratio'] = input_df['Down_payment'] / input_df['Property_value']
            
            print(f"After feature engineering: LTV={input_df['LTV'].iloc[0]:.3f}, EMI_to_income={input_df['EMI_to_income'].iloc[0]:.3f}")
            
            # One-hot encode Employment_type (matching training code)
            input_df = pd.get_dummies(input_df, columns=['Employment_type'], drop_first=True)
            
            print(f"Final prepared dataframe shape: {input_df.shape}")
            return input_df
            
        except Exception as e:
            print(f"Error in prepare_model_input: {e}")
            raise e
    
    def predict_loan(self, user_input: Dict[str, Any]) -> tuple:
        """Predict home loan amount and interest rate using XGBoost models"""
        try:
            print(f"Home Loan Prediction - Input: {user_input}")
            
            # Prepare input data with feature engineering
            input_df = self.prepare_model_input(user_input)
            print(f"Prepared input shape: {input_df.shape}")
            print(f"Input columns: {list(input_df.columns)}")
            
            # Try to use actual ML models if available
            if self.models.get("loan_amount_model") and self.models.get("interest_rate_model"):
                try:
                    print("Using ML models for prediction...")
                    # Get the training columns from the first model to align features
                    loan_model = self.models["loan_amount_model"]
                    rate_model = self.models["interest_rate_model"]
                    
                    # Ensure input has all required columns (fill missing with 0)
                    if hasattr(loan_model, 'feature_names_in_'):
                        required_cols = loan_model.feature_names_in_
                        input_df = input_df.reindex(columns=required_cols, fill_value=0)
                        print(f"Aligned to model columns: {len(required_cols)} features")
                    
                    # Make predictions
                    predicted_loan = loan_model.predict(input_df)[0]
                    predicted_rate = rate_model.predict(input_df)[0]
                    
                    print(f"ML Prediction - Loan: Rs.{predicted_loan:,.0f}, Rate: {predicted_rate:.2f}%")
                    return round(float(predicted_loan), 0), round(float(predicted_rate), 2)
                    
                except Exception as e:
                    print(f"🏠 Model prediction error: {e}")
                    raise Exception(f"ML model prediction failed: {str(e)}")
            else:
                raise Exception("ML models not loaded. Please ensure loan_amount_model.pkl and interest_rate_model.pkl are available in models/home_loan_models/")
            
        except Exception as e:
            print(f"🏠 Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Home loan prediction failed: {str(e)}")