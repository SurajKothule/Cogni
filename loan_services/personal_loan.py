from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import joblib
import re
import json
from .base_loan import BaseLoanService

class PersonalLoanService(BaseLoanService):
    """Personal Loan Service with ML Model Integration"""
    
    def get_required_fields(self) -> List[str]:
        return [
            # Customer Contact Information
            "Customer_Name",
            "Customer_Email", 
            "Customer_Phone",
            # Model Prediction Fields
            "Age",
            "Employment_Type",
            "Employment_Duration_Years",
            "Annual_Income",
            "CIBIL_Score",
            "Existing_EMIs",
            "Loan_Term_Years",
            "Expected_Loan_Amount",  # Added for frontend compatibility
        ]
    
    def get_model_files(self) -> Dict[str, str]:
        return {
            "personal_loan_model": "personal_loan_model.pkl",
        }
    
    def get_system_prompt(self) -> str:
        return """You are a friendly and professional personal loan advisor chatbot.

Your task is to systematically collect the following information from users through natural conversation:

Required Customer Information:
- Customer_Name (full name)
- Customer_Email (email address)
- Customer_Phone (10-digit phone number)

Required Loan Information:
- Age (21-65 years)
- Employment_Type: exactly one of ["Self-Employed", "Salaried"]
- Employment_Duration_Years (how many years working in current employment type)
- Annual_Income (yearly income in INR)
- CIBIL_Score (credit score, 300-900, minimum 650 recommended)
- Existing_EMIs (current monthly EMI obligations in INR, can be 0)
- Loan_Term_Years (repayment period in years, typically 1-7)
- Expected_Loan_Amount (desired loan amount in INR)

Guidelines:
1) Be conversational and friendly, like a helpful loan specialist.
2) Ask 1-2 related questions at a time to avoid overwhelming the customer.
3) Provide brief explanations when needed (e.g., "CIBIL score is your credit score").
4) If user provides partial info, acknowledge it positively and ask for missing details.
5) Validate responses and ask for clarification if unclear.
6) For categorical fields, ensure exact match with the specified options.
7) When you have ALL information, say exactly: INFORMATION_COMPLETE
8) Do NOT provide loan predictions - only collect information professionally.

Start by introducing yourself as a personal loan specialist."""
    
    def get_fallback_greeting(self) -> str:
        return "Hello! I'm a personal loan specialist here to help you with your loan application. Let's start with your full name - what should I call you?"
    
    def get_extraction_prompt(self, user_text: str, conversation: List[Dict[str, str]]) -> str:
        return f"""
Based on the conversation history and the user's latest response, extract any personal loan-related information.

Conversation so far: {conversation[-3:] if len(conversation) > 3 else conversation}

User's latest response: "{user_text}"

Extract information for these fields (only if clearly mentioned):

Customer Information:
- Customer_Name: full name as string
- Customer_Email: email address as string  
- Customer_Phone: 10-digit phone number as string (remove +91, spaces, dashes)

Loan Information:
- Age: number (21-65)
- Employment_Type: exactly one of ["Self-Employed", "Salaried"]
- Employment_Duration_Years: number (years in current employment type)
- Annual_Income: number in INR (yearly income, must be positive)
- CIBIL_Score: number (300-900, minimum 650 recommended)
- Existing_EMIs: number in INR (current monthly EMI obligations, 0 if none)
- Loan_Term_Years: number (years, typically 1-7)
- Expected_Loan_Amount: number in INR (desired loan amount)

Important:
- For Employment_Type, map variations like "self employed", "salaried employee" to exact options
- Convert lakhs/crores to actual numbers (e.g., "12 lakhs annual" = 1200000)
- For Employment_Duration_Years, ask about years in current employment type, not total experience
- Extract only information that is clearly stated

Return ONLY a JSON object with the extracted fields. If no information is found, return empty JSON {{}}.
Example: {{"Customer_Name": "John Doe", "Age": 35, "Employment_Type": "Salaried", "Annual_Income": 1200000, "Employment_Duration_Years": 12}}
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
                if 21 <= age <= 65:  # Personal loan specific age range
                    extracted['Age'] = age
                    break
        
        # Context-aware age extraction
        if not extracted.get('Age') and 'age' in last_assistant_msg:
            age_match = re.search(r'\b(\d{1,2})\b', user_text)
            if age_match:
                age = int(age_match.group(1))
                if 21 <= age <= 65:
                    extracted['Age'] = age
        
        # 5. PERSONAL LOAN SPECIFIC FIELDS
        
        # Annual Income extraction
        income_patterns = [
            r'(?:annual.*?income|yearly.*?salary|earn.*?yearly).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:annual.*?income|yearly.*?salary)',
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount >= 200000:  # Minimum reasonable income
                    extracted['Annual_Income'] = amount
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
        
        # Employment Type extraction
        employment_mapping = {
            'self employed': 'Self-Employed', 'self-employed': 'Self-Employed', 'business': 'Self-Employed', 'entrepreneur': 'Self-Employed',
            'salaried': 'Salaried', 'employee': 'Salaried', 'job': 'Salaried', 'working': 'Salaried'
        }
        
        for keyword, employment in employment_mapping.items():
            if keyword in text_lower:
                extracted['Employment_Type'] = employment
                break
        
        # Employment Duration extraction
        duration_patterns = [
            r'(?:working|employed|experience).*?(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?).*?(?:working|employed|experience)',
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text_lower)
            if match:
                duration = int(match.group(1))
                if 1 <= duration <= 45:  # Reasonable employment duration
                    extracted['Employment_Duration_Years'] = duration
                    break
        
        # Existing EMIs extraction
        emi_patterns = [
            r'(?:existing.*?emi|current.*?emi|monthly.*?payment).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:emi.*?payment)',
        ]
        
        for pattern in emi_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount >= 0:
                    extracted['Existing_EMIs'] = amount
                    break
        
        # Loan Term extraction
        term_patterns = [
            r'(?:term|duration|years?)\s*(?:is\s*)?(?::|=)?\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?).*?(?:term|duration)',
        ]
        
        for pattern in term_patterns:
            match = re.search(pattern, text_lower)
            if match:
                term = int(match.group(1))
                if 1 <= term <= 7:  # Personal loan specific term range
                    extracted['Loan_Term_Years'] = term
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
                if amount and amount >= 50000:  # Minimum loan amount
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
        """Validate individual field values with strict eligibility criteria"""
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
                if age < 21:
                    return False, "INELIGIBLE: You must be at least 21 years old to apply for a personal loan. Unfortunately, we cannot process your application at this time."
                elif age > 65:
                    return False, "INELIGIBLE: Personal loans are available only for applicants up to 65 years of age. Unfortunately, we cannot process your application at this time."
                return True, ""
                
            elif field_name == "CIBIL_Score":
                cibil = float(value)
                if cibil < 650:
                    return False, "INELIGIBLE: A minimum CIBIL score of 650 is required for personal loan approval. Your current score does not meet our eligibility criteria."
                elif not (300 <= cibil <= 900):
                    return False, "Please provide a valid CIBIL score between 300 and 900. Could you check and confirm your credit score?"
                return True, ""
                
            elif field_name == "Employment_Duration_Years":
                duration = float(value)
                if duration < 0:
                    return False, "INELIGIBLE: Employment duration cannot be negative. Please provide valid employment experience."
                elif duration < 1:
                    return False, "INELIGIBLE: You must have at least 1 year of employment experience to qualify for a personal loan."
                elif duration > 45:
                    return False, "Employment duration seems unusually high. Could you please confirm how many years you've been in your current employment type?"
                return True, ""
                
            elif field_name == "Annual_Income":
                income = float(value)
                if income <= 0:
                    return False, "Annual income must be a positive amount. Please provide your yearly income."
                elif income < 200000:  # Minimum 2 lakhs per year
                    return False, "INELIGIBLE: Minimum annual income of ₹2,00,000 is required for personal loan eligibility."
                elif income > 50000000:  # Maximum 5 crores
                    return False, "Please verify your annual income. The amount seems unusually high. Could you confirm?"
                return True, ""
                
            elif field_name == "Employment_Type":
                valid_types = ["Self-Employed", "Salaried"]
                if value not in valid_types:
                    return False, f"Please select your employment type from: {', '.join(valid_types)}. Which category describes your employment?"
                return True, ""
                
            elif field_name == "Loan_Term_Years":
                term = float(value)
                if not (1 <= term <= 7):
                    return False, "Loan term must be between 1 and 7 years. Please specify your preferred repayment period."
                return True, ""
                
            elif field_name == "Expected_Loan_Amount":
                amount = float(value)
                if amount <= 0:
                    return False, "Loan amount must be a positive value. Please specify your loan requirement."
                elif amount < 50000:
                    return False, "Minimum loan amount is ₹50,000. Please specify an amount of at least ₹50,000."
                elif amount > 2000000:
                    return False, "Maximum loan amount is ₹20,00,000. Please specify an amount within this limit."
                return True, ""
                
            elif field_name == "Existing_EMIs":
                amount = float(value)
                if amount < 0:
                    return False, "EMI amount cannot be negative. Please provide your current monthly EMI obligations (enter 0 if none)."
                return True, ""
                
            return True, ""
            
        except (ValueError, TypeError):
            field_display = field_name.replace('_', ' ').lower()
            return False, f"Please provide a valid {field_display} in the correct format."

    def calculate_debt_to_income_ratio(self, annual_income: float, existing_emi: float, proposed_emi: float) -> float:
        """Calculate Debt-to-Income ratio"""
        monthly_income = annual_income / 12
        total_debt = existing_emi + proposed_emi
        return (total_debt / monthly_income) * 100 if monthly_income > 0 else 0

    def prepare_model_input(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data for the personal loan model based on your exact model structure"""
        try:
            # Create input dataframe matching your model's expected features
            # Based on your code: Age, Employment_Type, Employment_Duration_Years, Annual_Income, CIBIL_Score, Existing_EMIs, Loan_Term_Years
            
            input_data = {
                'Age': float(user_input['Age']),
                'Employment_Type': user_input['Employment_Type'],  # Will be encoded later
                'Employment_Duration_Years': float(user_input['Employment_Duration_Years']),
                'Annual_Income': float(user_input['Annual_Income']),
                'CIBIL_Score': float(user_input['CIBIL_Score']),
                'Existing_EMIs': float(user_input.get('Existing_EMIs', 0)),
                'Loan_Term_Years': float(user_input['Loan_Term_Years'])
            }
            
            print(f"Personal Loan Input data prepared: {input_data}")
            
            input_df = pd.DataFrame([input_data])
            
            print(f"Prepared input dataframe shape: {input_df.shape}")
            print(f"Input columns: {list(input_df.columns)}")
            
            return input_df
            
        except Exception as e:
            print(f"Error in prepare_model_input: {e}")
            raise e
    
    def predict_loan(self, user_input: Dict[str, Any]) -> tuple:
        """Predict personal loan amount and interest rate using ML model"""
        try:
            print(f"Personal Loan Prediction - Input: {user_input}")
            
            # Prepare input data
            input_df = self.prepare_model_input(user_input)
            print(f"Prepared input shape: {input_df.shape}")
            print(f"Input columns: {list(input_df.columns)}")
            
            # Try to use actual ML model if available
            if self.models.get("personal_loan_model"):
                try:
                    print("Using ML model for prediction...")
                    
                    # Load the model package - matching your structure exactly
                    package = self.models["personal_loan_model"]
                    model = package["model"]
                    scaler = package["scaler"]
                    le = package["encoder"]  # Label encoder for Employment_Type
                    features = package["features"]  # Feature order
                    
                    print("Model components loaded successfully")
                    print(f"Expected features: {features}")
                    
                    # Prepare data for prediction
                    df_input = input_df.copy()
                    
                    # Encode Employment_Type using your label encoder
                    df_input["Employment_Type"] = le.transform(df_input["Employment_Type"])
                    print(f"After encoding Employment_Type: {df_input['Employment_Type'].values}")
                    
                    # Select features in the correct order
                    df_input = df_input[features]
                    print(f"After feature selection: {list(df_input.columns)}")
                    
                    # Scale features using your scaler
                    df_scaled = scaler.transform(df_input)
                    print(f"Scaled features shape: {df_scaled.shape}")
                    
                    # Make predictions
                    prediction = model.predict(df_scaled)[0]
                    print(f"Raw predictions: {prediction}")
                    
                    # Handle predictions as per your structure
                    # prediction[0] = log-transformed loan amount
                    # prediction[1] = interest rate
                    loan_amount = np.expm1(prediction[0])  # inverse log transformation
                    interest_rate = float(prediction[1])
                    
                    print(f"After transformation - Loan: Rs.{loan_amount:,.0f}, Rate: {interest_rate:.2f}%")
                    
                    # Ensure reasonable bounds
                    loan_amount = max(50000, min(2000000, loan_amount))  # Between 50k and 20L
                    interest_rate = max(8.0, min(18.0, interest_rate))   # Between 8% and 18%
                    
                    print(f"Final ML Prediction - Loan: Rs.{loan_amount:,.0f}, Rate: {interest_rate:.2f}%")
                    return round(float(loan_amount), 0), round(float(interest_rate), 2)
                    
                except Exception as e:
                    print(f"Model prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"ML model prediction failed: {str(e)}")
            else:
                raise Exception("Personal loan ML model not available. Cannot process loan prediction.")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Personal loan prediction failed: {str(e)}")