from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import joblib
import re
import json
from .base_loan import BaseLoanService

class GoldLoanService(BaseLoanService):
    """Gold Loan Service with ML Model Integration"""
    
    def get_required_fields(self) -> List[str]:
        return [
            # Customer Contact Information
            "Customer_Name",
            "Customer_Email", 
            "Customer_Phone",
            # Model Prediction Fields
            "Age",
            "Annual_Income",
            "CIBIL_Score",
            "Occupation",
            "Gold_Value",
            "Loan_Amount",
            "Loan_Tenure",
        ]
    
    def get_model_files(self) -> Dict[str, str]:
        return {
            "gold_loan_model": "gold_loan_model.pkl",
        }
    
    def get_system_prompt(self) -> str:
        return """You are a friendly and professional gold loan advisor chatbot.

Your task is to systematically collect the following information from users through natural conversation:

Required Customer Information:
- Customer_Name (full name)
- Customer_Email (email address)
- Customer_Phone (10-digit phone number)

Required Loan Information:
- Age (21-75 years for gold loans)
- Annual_Income (yearly income in INR)
- CIBIL_Score (credit score, 300-900, minimum 600 for gold loans)
- Occupation: exactly one of ["Salaried", "Retired", "Business", "Self-employed"]
- Gold_Value (current market value of your gold in INR - we'll assess the actual value during verification)
- Loan_Amount (desired loan amount in INR)
- Loan_Tenure (repayment period in years, typically 1-3 for gold loans)

Guidelines:
1) Be conversational and friendly, like a helpful gold loan specialist.
2) Ask 1-2 related questions at a time to avoid overwhelming the customer.
3) For Gold_Value, ask for the approximate current market value of their gold in INR.
4) Do NOT ask for gold weight, purity, or rate per gram - only ask for the total gold value.
5) If user provides partial info, acknowledge it positively and ask for missing details.
6) Validate responses and ask for clarification if unclear.
7) For Occupation, provide the exact options: Salaried, Retired, Business, Self-employed.
8) When you have ALL information, say exactly: INFORMATION_COMPLETE
9) Do NOT provide loan predictions - only collect information professionally.

Start by introducing yourself as a gold loan specialist."""
    
    def get_fallback_greeting(self) -> str:
        return "Hello! I'm a gold loan specialist here to help you with your gold loan application. Gold loans offer quick financing against your gold jewelry. Let's start with your full name - what should I call you?"
    
    def get_extraction_prompt(self, user_text: str, conversation: List[Dict[str, str]]) -> str:
        return f"""
Based on the conversation history and the user's latest response, extract any gold loan-related information.

Conversation so far: {conversation[-3:] if len(conversation) > 3 else conversation}

User's latest response: "{user_text}"

Extract information for these fields (only if clearly mentioned):

Customer Information:
- Customer_Name: full name as string
- Customer_Email: email address as string  
- Customer_Phone: 10-digit phone number as string (remove +91, spaces, dashes)

Loan Information:
- Age: number (21-75)
- Annual_Income: number in INR (yearly income, must be positive)
- CIBIL_Score: number (300-900, minimum 600 for gold loans)
- Occupation: exactly one of ["Salaried", "Retired", "Business", "Self-employed"]
- Gold_Value: number (current market value of gold in INR)
- Loan_Amount: number (desired loan amount in INR)
- Loan_Tenure: number (years, typically 1-3 for gold loans)

Important:
- For Occupation, map variations like "salaried employee", "business owner", "retired person" to exact options
- Convert lakhs/crores to actual numbers (e.g., "5 lakhs income" = 500000)
- Extract only information that is clearly stated
- Do NOT extract Gold_Weight, Gold_Purity, or Gold_Rate_Per_Gram - only Gold_Value

Return ONLY a JSON object with the extracted fields. If no information is found, return empty JSON {{}}.
Example: {{"Customer_Name": "John Doe", "Age": 45, "Annual_Income": 900000, "Occupation": "Salaried", "Gold_Value": 400000, "Loan_Amount": 300000, "Loan_Tenure": 2}}
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
                if 21 <= age <= 75:  # Gold loan specific age range
                    extracted['Age'] = age
                    break
        
        # Context-aware age extraction
        if not extracted.get('Age') and 'age' in last_assistant_msg:
            age_match = re.search(r'\b(\d{1,2})\b', user_text)
            if age_match:
                age = int(age_match.group(1))
                if 21 <= age <= 75:
                    extracted['Age'] = age
        
        # 5. GOLD LOAN SPECIFIC FIELDS
        
        # Annual Income extraction
        income_patterns = [
            r'(?:annual.*?income|salary|earn).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:annual.*?income|salary|yearly)',
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount >= 180000:  # Minimum reasonable income
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
        
        # Occupation extraction
        occupation_mapping = {
            'salaried': 'Salaried', 'employee': 'Salaried', 'job': 'Salaried', 'working': 'Salaried',
            'retired': 'Retired', 'pension': 'Retired', 'senior': 'Retired',
            'business': 'Business', 'businessman': 'Business', 'trader': 'Business', 'merchant': 'Business',
            'self employed': 'Self-employed', 'self-employed': 'Self-employed', 'freelance': 'Self-employed', 'consultant': 'Self-employed'
        }
        
        for keyword, occupation in occupation_mapping.items():
            if keyword in text_lower:
                extracted['Occupation'] = occupation
                break
        
        # Gold Value extraction
        gold_value_patterns = [
            r'(?:gold.*?value|gold.*?worth|jewelry.*?value).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:gold.*?value|gold.*?worth)',
        ]
        
        for pattern in gold_value_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount >= 10000:  # Minimum reasonable gold value
                    extracted['Gold_Value'] = amount
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
                if amount and amount >= 5000:  # Minimum loan amount
                    extracted['Loan_Amount'] = amount
                    break
        
        # Loan Tenure extraction
        tenure_patterns = [
            r'(?:tenure|duration|years?)\s*(?:is\s*)?(?::|=)?\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?).*?(?:tenure|duration)',
        ]
        
        for pattern in tenure_patterns:
            match = re.search(pattern, text_lower)
            if match:
                tenure = int(match.group(1))
                if 1 <= tenure <= 3:  # Gold loan specific tenure range
                    extracted['Loan_Tenure'] = tenure
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
                    return False, "INELIGIBLE: You must be at least 21 years old to apply for a gold loan. Unfortunately, we cannot process your application at this time."
                elif age > 75:
                    return False, "INELIGIBLE: Gold loans are available only for applicants up to 75 years of age. Unfortunately, we cannot process your application at this time."
                return True, ""
                
            elif field_name == "CIBIL_Score":
                cibil = float(value)
                if cibil < 600:
                    return False, "INELIGIBLE: A minimum CIBIL score of 600 is required for gold loan approval. Your current score does not meet our eligibility criteria."
                elif not (300 <= cibil <= 900):
                    return False, "Please provide a valid CIBIL score between 300 and 900. Could you check and confirm your credit score?"
                return True, ""
                
            elif field_name == "Occupation":
                valid_occupations = ["Salaried", "Retired", "Business", "Self-employed"]
                if value not in valid_occupations:
                    return False, f"Please select your occupation from: {', '.join(valid_occupations)}. Which category best describes your occupation?"
                return True, ""
                
            elif field_name == "Annual_Income":
                income = float(value)
                if income <= 0:
                    return False, "Annual income must be a positive amount. Please provide your yearly income."
                elif income < 180000:  # Minimum 1.8 lakhs per year
                    return False, "INELIGIBLE: Minimum annual income of ₹1,80,000 is required for gold loan eligibility."
                elif income > 60000000:  # Maximum 6 crores per year
                    return False, "Please verify your annual income. The amount seems unusually high. Could you confirm?"
                return True, ""
                
            elif field_name == "Gold_Value":
                value_amount = float(value)
                if value_amount <= 0:
                    return False, "Gold value must be a positive amount. Please provide the current market value of your gold in INR."
                elif value_amount < 10000:  # Minimum 10k gold value
                    return False, "INELIGIBLE: Minimum gold value of ₹10,000 is required for gold loan eligibility."
                elif value_amount > 50000000:  # Maximum 5 crores
                    return False, "Please verify your gold value. The amount seems unusually high. Could you confirm the current market value?"
                return True, ""
                
            elif field_name == "Loan_Amount":
                amount = float(value)
                if amount <= 0:
                    return False, "Loan amount must be a positive amount. Please provide your desired loan amount in INR."
                elif amount < 5000:  # Minimum 5k loan
                    return False, "INELIGIBLE: Minimum loan amount of ₹5,000 is required."
                elif amount > 10000000:  # Maximum 1 crore
                    return False, "Please verify your loan amount. The amount seems unusually high for a gold loan."
                return True, ""
                
            elif field_name == "Loan_Tenure":
                tenure = float(value)
                if tenure < 1:
                    return False, "INELIGIBLE: Gold loan tenure must be at least 1 year. Please specify a tenure between 1 and 3 years."
                elif tenure > 3:
                    return False, "INELIGIBLE: Gold loan tenure cannot exceed 3 years. Please specify a tenure between 1 and 3 years."
                return True, ""
                
            return True, ""
            
        except (ValueError, TypeError):
            field_display = field_name.replace('_', ' ').lower()
            return False, f"Please provide a valid {field_display} in the correct format."

    def prepare_model_input(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data for the gold loan model"""
        try:
            # Convert Annual Income to Monthly Income
            monthly_income = float(user_input['Annual_Income']) / 12
            
            # Create input dataframe matching your model's expected features
            # Model expects: ['Age', 'Occupation', 'Monthly_Income', 'CIBIL_Score', 'Gold_Value', 'Existing_EMI', 'Loan_Tenure_Years']
            input_data = {
                'Age': float(user_input['Age']),
                'Occupation': user_input['Occupation'],  # Will be encoded later
                'Monthly_Income': monthly_income,
                'CIBIL_Score': float(user_input['CIBIL_Score']),
                'Gold_Value': float(user_input['Gold_Value']),
                'Existing_EMI': 0.0,  # Default to 0 since we don't collect this anymore
                'Loan_Tenure_Years': float(user_input['Loan_Tenure'])
            }
            
            print(f"Gold Loan Input data prepared: {input_data}")
            
            input_df = pd.DataFrame([input_data])
            
            print(f"Prepared input dataframe shape: {input_df.shape}")
            print(f"Input columns: {list(input_df.columns)}")
            
            return input_df
            
        except Exception as e:
            print(f"Error in prepare_model_input: {e}")
            raise e
    
    def predict_loan(self, user_input: Dict[str, Any]) -> tuple:
        """Predict gold loan amount and interest rate using ML model"""
        try:
            print(f"Gold Loan Prediction - Input: {user_input}")
            
            # Prepare input data
            input_df = self.prepare_model_input(user_input)
            print(f"Prepared input shape: {input_df.shape}")
            print(f"Input columns: {list(input_df.columns)}")
            
            # Try to use actual ML model if available
            if self.models.get("gold_loan_model"):
                try:
                    print("Using ML model for prediction...")
                    
                    # Load the model package - matching your structure exactly
                    package = self.models["gold_loan_model"]
                    model = package["model"]
                    scaler = package["scaler"]
                    encoder = package["encoder"]  # Label encoder for Occupation
                    features = package["features"]  # Feature order
                    targets = package["targets"]  # Target names
                    
                    print("Model components loaded successfully")
                    print(f"Expected features: {features}")
                    print(f"Target variables: {targets}")
                    
                    # Prepare data for prediction
                    df_input = input_df.copy()
                    
                    # Encode Occupation using your label encoder
                    df_input["Occupation"] = encoder.transform(df_input["Occupation"])
                    print(f"After encoding Occupation: {df_input['Occupation'].values}")
                    
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
                    # prediction[0] = Loan_Amount
                    # prediction[1] = Rate_of_Interest
                    loan_amount = float(prediction[0])
                    interest_rate = float(prediction[1])
                    
                    print(f"ML Prediction - Loan: Rs.{loan_amount:,.0f}, Rate: {interest_rate:.2f}%")
                    
                    # Ensure reasonable bounds for gold loans
                    # Gold loans typically offer 70-80% of gold value
                    max_loan_based_on_gold = float(user_input['Gold_Value']) * 0.8
                    loan_amount = min(loan_amount, max_loan_based_on_gold)
                    loan_amount = max(loan_amount, 5000)    # Min 5k
                    interest_rate = max(8.0, min(24.0, interest_rate))   # Between 8% and 24%
                    
                    print(f"Final ML Prediction - Loan: Rs.{loan_amount:,.0f}, Rate: {interest_rate:.2f}%")
                    return round(float(loan_amount), 0), round(float(interest_rate), 2)
                    
                except Exception as e:
                    print(f"Model prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"ML model prediction failed: {str(e)}")
            else:
                raise Exception("Gold loan ML model not available. Cannot process loan prediction.")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Gold loan prediction failed: {str(e)}")