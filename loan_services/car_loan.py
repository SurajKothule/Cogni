from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import pickle
import re
import json
from .base_loan import BaseLoanService

class CarLoanService(BaseLoanService):
    """Car Loan Service with ML Model Integration"""
    
    def get_required_fields(self) -> List[str]:
        return [
            # Customer Contact Information
            "Customer_Name",
            "Customer_Email", 
            "Customer_Phone",
            # Model Prediction Fields
            "Age",
            "applicant_annual_salary",
            "Coapplicant_Annual_Income",
            "CIBIL",
            "Car_Type",
            "down_payment_percent",
            "Tenure",
            "loan_amount",
        ]
    
    def get_model_files(self) -> Dict[str, str]:
        return {
            "car_loan_model": "car_loan_models.pkl",
        }
    
    def get_system_prompt(self) -> str:
        return """You are a friendly and professional car loan advisor chatbot.

Your task is to systematically collect the following information from users through natural conversation:

Required Customer Information:
- Customer_Name (full name)
- Customer_Email (email address)
- Customer_Phone (10-digit phone number)

Required Car Loan Information:
- Age (applicant's age in years, 18-80)
- applicant_annual_salary (primary applicant's yearly salary in INR)
- Coapplicant_Annual_Income (co-applicant's yearly income in INR, can be 0 if no co-applicant)
- CIBIL (credit score, 300-900, minimum 650 for car loans)
- Car_Type: exactly one of ["Sedan", "SUV", "Hatchback", "Coupe"]
- down_payment_percent (down payment as percentage, 10-50%)
- Tenure (loan repayment period in years, typically 1-7 years)
- loan_amount (desired loan amount in INR)

Guidelines:
1) Be conversational and friendly, like a helpful car loan specialist.
2) Ask 1-2 related questions at a time to avoid overwhelming the customer.
3) Provide brief explanations when needed (e.g., "Down payment is the upfront amount you pay").
4) If user provides partial info, acknowledge it positively and ask for missing details.
5) Validate responses and ask for clarification if unclear.
6) For categorical fields, provide the exact options to choose from.
7) When you have ALL information, say exactly: INFORMATION_COMPLETE
8) Do NOT provide loan predictions - only collect information professionally.
9) VARY your responses - don't use the same greeting or acknowledgment repeatedly.
10) Make the conversation engaging and natural.

Start by introducing yourself as a car loan specialist."""
    
    def get_fallback_greeting(self) -> str:
        greetings = [
            "Hello! I'm a car loan specialist here to help you with your car loan application. Car loans can help you purchase your dream vehicle with flexible repayment options. Let's start with your full name - what should I call you?",
            "Hi there! I'm excited to help you with your car loan journey. Getting the right financing can make your dream car a reality. May I know your name to get started?",
            "Welcome! As your car loan advisor, I'm here to guide you through the application process. Let's begin with your name - what would you like me to call you?",
            "Greetings! I specialize in car loans and I'm here to assist you in finding the perfect financing option. Could you share your name with me to get started?"
        ]
        return np.random.choice(greetings)
    
    def get_conversation_prompt(self, current_field: str, collected_data: Dict[str, Any], conversation: List[Dict[str, str]]) -> str:
        """Generate dynamic conversation prompts using OpenAI"""
        field_descriptions = {
            "Customer_Name": "full name",
            "Customer_Email": "email address",
            "Customer_Phone": "10-digit phone number",
            "Age": "age in years (18-80)",
            "applicant_annual_salary": "annual salary in INR",
            "Coapplicant_Annual_Income": "co-applicant's annual income in INR (0 if none)",
            "CIBIL": "CIBIL credit score (300-900)",
            "Car_Type": "car type (Sedan, SUV, Hatchback, or Coupe)",
            "down_payment_percent": "down payment percentage (10-50%)",
            "Tenure": "loan tenure in years (1-7)",
            "loan_amount": "desired loan amount in INR"
        }
        
        prompt = f"""
You are a friendly car loan specialist having a natural conversation with a customer.

Collected information so far:
{json.dumps(collected_data, indent=2)}

Current conversation history (last 3 messages):
{conversation[-3:] if len(conversation) > 3 else conversation}

Next field to collect: {current_field} ({field_descriptions[current_field]})

Guidelines:
1. Be warm, professional, and engaging
2. Acknowledge what you've collected so far naturally
3. Ask for the next field in a conversational way
4. Vary your phrasing - don't repeat the same patterns
5. Provide context if needed (e.g., explain what CIBIL score is)
6. Keep it concise (1-2 sentences)
7. Make it feel like a natural human conversation

Generate ONLY the next message to ask for {current_field}. Do not include any other text.
""".strip()
        
        return prompt
    
    def generate_conversational_message(self, current_field: str, collected_data: Dict[str, Any], conversation: List[Dict[str, str]]) -> str:
        """Generate dynamic conversational messages using OpenAI"""
        if not self.client:
            # Fallback to simple messages if OpenAI is not available
            fallback_messages = {
                "Customer_Name": "Could you please tell me your full name?",
                "Customer_Email": "May I have your email address?",
                "Customer_Phone": "What's your 10-digit phone number?",
                "Age": "How old are you?",
                "applicant_annual_salary": "What's your annual salary?",
                "Coapplicant_Annual_Income": "Does your co-applicant have any annual income?",
                "CIBIL": "What's your CIBIL credit score?",
                "Car_Type": "What type of car are you interested in?",
                "down_payment_percent": "What percentage down payment can you make?",
                "Tenure": "How many years would you like for the loan tenure?",
                "loan_amount": "How much loan amount are you looking for?"
            }
            return fallback_messages.get(current_field, f"Could you provide your {current_field.replace('_', ' ').lower()}?")
        
        try:
            prompt = self.get_conversation_prompt(current_field, collected_data, conversation)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # Higher temperature for more varied responses
                max_tokens=100
            )
            
            message = response.choices[0].message.content.strip()
            return message
            
        except Exception as e:
            print(f"OpenAI conversation generation failed: {e}")
            # Fallback to simple message
            return f"Could you please provide your {current_field.replace('_', ' ').lower()}?"

    def get_extraction_prompt(self, user_text: str, conversation: List[Dict[str, str]]) -> str:
        return f"""
Based on the conversation history and the user's latest response, extract any car loan-related information.

Conversation so far: {conversation[-3:] if len(conversation) > 3 else conversation}

User's latest response: "{user_text}"

Extract information for these fields (only if clearly mentioned):

Customer Information:
- Customer_Name: full name as string
- Customer_Email: email address as string  
- Customer_Phone: 10-digit phone number as string (remove +91, spaces, dashes)

Car Loan Information:
- Age: number (applicant's age in years, 18-80)
- applicant_annual_salary: number in INR (primary applicant's yearly salary, must be positive)
- Coapplicant_Annual_Income: number in INR (co-applicant's yearly income, 0 if none)
- CIBIL: number (300-900, minimum 650 for car loans)
- Car_Type: exactly one of ["Sedan", "SUV", "Hatchback", "Coupe"]
- down_payment_percent: number (down payment percentage, 10-50)
- Tenure: number (loan tenure in years, 1-7)
- loan_amount: number in INR (desired loan amount, must be positive)

Important conversion rules:
- Convert lakhs/crores to actual numbers: "20 lakh" = 2000000, "5 lakh" = 500000, "1.5 crore" = 15000000
- For Car_Type, map variations like "sedan car", "SUV vehicle" to exact options
- Extract only information that is clearly stated

Return ONLY a JSON object with the extracted fields. If no information is found, return empty JSON {{}}.
Example: {{"Customer_Name": "John Doe", "Age": 30, "applicant_annual_salary": 800000, "Car_Type": "Sedan", "CIBIL": 750}}
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
                if 18 <= age <= 80:
                    extracted['Age'] = age
                    break
        
        # Context-aware age extraction
        if not extracted.get('Age') and 'age' in last_assistant_msg:
            age_match = re.search(r'\b(\d{1,2})\b', user_text)
            if age_match:
                age = int(age_match.group(1))
                if 18 <= age <= 80:
                    extracted['Age'] = age
        
        # 5. CAR LOAN SPECIFIC FIELDS
        
        # Salary/Income extraction with conversion
        salary_patterns = [
            r'(?:salary|income|earn).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:salary|income|monthly)',
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount >= 300000:
                    if 'co' in last_assistant_msg or 'coapplicant' in last_assistant_msg:
                        extracted['Coapplicant_Annual_Income'] = amount
                    else:
                        extracted['applicant_annual_salary'] = amount
                    break
        
        # CIBIL score
        cibil_patterns = [
            r'(?:cibil|credit.*?score)\s*(?:is\s*)?(?::|=)?\s*(\d{3})',
            r'(\d{3})\s*(?:cibil|credit.*?score)',
        ]
        
        for pattern in cibil_patterns:
            match = re.search(pattern, text_lower)
            if match:
                score = int(match.group(1))
                if 300 <= score <= 900:
                    extracted['CIBIL'] = score
                    break
        
        # Car type extraction
        car_type_mapping = {
            'sedan': 'Sedan', 'suv': 'SUV', 'hatchback': 'Hatchback', 'coupe': 'Coupe',
            'maruti': 'Hatchback', 'hyundai': 'Sedan', 'tata': 'SUV', 'honda': 'Sedan',
            'swift': 'Hatchback', 'city': 'Sedan', 'creta': 'SUV', 'nexon': 'SUV'
        }
        
        for keyword, car_type in car_type_mapping.items():
            if keyword in text_lower:
                extracted['Car_Type'] = car_type
                break
        
        # Down payment percentage
        down_payment_patterns = [
            r'(?:down.*?payment|advance).*?(\d+)\s*(?:%|percent|percentage)',
            r'(\d+)\s*(?:%|percent|percentage).*?(?:down.*?payment|advance)',
        ]
        
        for pattern in down_payment_patterns:
            match = re.search(pattern, text_lower)
            if match:
                percent = int(match.group(1))
                if 10 <= percent <= 50:
                    extracted['down_payment_percent'] = percent
                    break
        
        # Tenure extraction
        tenure_patterns = [
            r'(?:tenure|duration|period).*?(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?).*?(?:tenure|duration|period)',
        ]
        
        for pattern in tenure_patterns:
            match = re.search(pattern, text_lower)
            if match:
                tenure = int(match.group(1))
                if 1 <= tenure <= 7:
                    extracted['Tenure'] = tenure
                    break
        
        # Loan amount extraction
        loan_amount_patterns = [
            r'(?:loan.*?amount|need.*?loan|want.*?loan).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:loan.*?amount|need.*?loan)',
        ]
        
        for pattern in loan_amount_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount and amount >= 100000:
                    extracted['loan_amount'] = amount
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
                if age < 18:
                    return False, "INELIGIBLE: You must be at least 18 years old to apply for a car loan."
                elif age > 80:
                    return False, "INELIGIBLE: Maximum age limit for car loan is 80 years."
                return True, ""
                
            # Car loan specific validations
            elif field_name == "CIBIL":
                cibil = float(value)
                if cibil < 650:
                    return False, "INELIGIBLE: A minimum CIBIL score of 650 is required for car loan approval. Your current score does not meet our eligibility criteria."
                elif not (300 <= cibil <= 900):
                    return False, "Please provide a valid CIBIL score between 300 and 900. Could you check and confirm your credit score?"
                return True, ""
                
            elif field_name == "applicant_annual_salary":
                salary = float(value)
                if salary <= 0:
                    return False, "Annual salary must be a positive amount. Please provide your yearly salary."
                elif salary < 300000:
                    return False, "INELIGIBLE: Minimum annual salary of ₹3,00,000 is required for car loan eligibility."
                elif salary > 100000000:
                    return False, "Please verify your annual salary. The amount seems unusually high. Could you confirm your yearly income?"
                return True, ""
                
            elif field_name == "Coapplicant_Annual_Income":
                income = float(value)
                if income < 0:
                    return False, "Co-applicant income cannot be negative. Please provide the co-applicant's yearly income (enter 0 if no co-applicant)."
                elif income > 100000000:
                    return False, "Please verify the co-applicant's income. The amount seems unusually high."
                return True, ""
                
            elif field_name == "Car_Type":
                valid_types = ["Sedan", "SUV", "Hatchback", "Coupe"]
                if value not in valid_types:
                    return False, f"Please select your car type from: {', '.join(valid_types)}. Which type of car are you planning to purchase?"
                return True, ""
                
            elif field_name == "down_payment_percent":
                percent = float(value)
                if not (10 <= percent <= 50):
                    return False, "Down payment percentage must be between 10% and 50%. Please specify your down payment percentage."
                return True, ""
                
            elif field_name == "Tenure":
                tenure = float(value)
                if not (1 <= tenure <= 7):
                    return False, "Car loan tenure must be between 1 and 7 years. Please specify your preferred repayment period."
                return True, ""
                
            elif field_name == "loan_amount":
                amount = float(value)
                if amount <= 0:
                    return False, "Loan amount must be a positive amount. Please specify how much loan you need."
                elif amount < 100000:
                    return False, "INELIGIBLE: Minimum loan amount is ₹1,00,000 for car loans."
                elif amount > 50000000:
                    return False, "Please verify your loan requirement. The amount seems unusually high for a car loan. Could you confirm the loan amount needed?"
                return True, ""
                
            return True, ""
            
        except (ValueError, TypeError):
            field_display = field_name.replace('_', ' ').lower()
            return False, f"Please provide a valid {field_display} in the correct format."

    def prepare_model_input(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data for the car loan model"""
        try:
            # Map categorical values to model format
            car_type_map = {
                "Sedan": 0,
                "SUV": 1,
                "Hatchback": 2,
                "Coupe": 3
            }
            
            # Calculate Total_Annual_Income
            applicant_salary = float(user_input['applicant_annual_salary'])
            coapplicant_income = float(user_input.get('Coapplicant_Annual_Income', 0))
            total_annual_income = applicant_salary + coapplicant_income
            
            # Use default employment type (Salaried = 0) since we don't ask user
            default_employment_type = 0  # Salaried is most common
            
            # Create input dataframe matching your model's expected features
            input_data = {
                'applicant_annual_salary': applicant_salary,
                'Coapplicant_Annual_Income': coapplicant_income,
                'Total_Annual_Income': total_annual_income,
                'CIBIL': float(user_input['CIBIL']),
                'Employment_Type': default_employment_type,  # Default to Salaried
                'Car_Type': car_type_map[user_input['Car_Type']],
                'down_payment_percent': float(user_input['down_payment_percent']),
                'Tenure': float(user_input['Tenure']),
                'Age': float(user_input['Age'])
            }
            
            print(f"Car Loan Input data prepared: {input_data}")
            
            input_df = pd.DataFrame([input_data])
            
            print(f"Prepared input dataframe shape: {input_df.shape}")
            print(f"Input columns: {list(input_df.columns)}")
            
            return input_df
            
        except Exception as e:
            print(f"Error in prepare_model_input: {e}")
            raise e
    
    def predict_loan(self, user_input: Dict[str, Any]) -> tuple:
        """Predict car loan amount and interest rate using ML model"""
        try:
            print(f"Car Loan Prediction - Input: {user_input}")
            
            # Prepare input data
            input_df = self.prepare_model_input(user_input)
            print(f"Prepared input shape: {input_df.shape}")
            print(f"Input columns: {list(input_df.columns)}")
            
            # Try to use actual ML model if available
            if self.models.get("car_loan_model"):
                try:
                    print("Using ML model for prediction...")
                    
                    # Load the model bundle
                    bundle = self.models["car_loan_model"]
                    model_max_amt = bundle["model_max_amt"]
                    model_rate = bundle["model_rate"]
                    scaler = bundle["scaler"]
                    label_encoders = bundle["label_encoders"]
                    features = bundle["features"]
                    
                    print("Model components loaded successfully")
                    print(f"Expected features: {features}")
                    
                    # Prepare data for prediction - ensure correct column order
                    df_input = input_df[features]
                    print(f"After feature selection: {list(df_input.columns)}")
                    
                    # Scale the features
                    df_scaled = scaler.transform(df_input)
                    print(f"Data scaled successfully")
                    
                    # Make predictions
                    max_loan_amount = model_max_amt.predict(df_scaled)[0]
                    interest_rate = model_rate.predict(df_scaled)[0]
                    
                    print(f"Raw predictions - Max Loan: {max_loan_amount}, Interest Rate: {interest_rate}")
                    
                    # Ensure reasonable bounds for car loans
                    max_loan_amount = max(max_loan_amount, 100000)    # Min 1 lakh
                    max_loan_amount = min(max_loan_amount, 50000000)  # Max 5 crores
                    interest_rate = max(7.0, min(20.0, interest_rate)) # Between 7% and 20%
                    
                    print(f"Final ML Prediction - Loan: Rs.{max_loan_amount:,.0f}, Rate: {interest_rate:.2f}%")
                    return round(float(max_loan_amount), 0), round(float(interest_rate), 2)
                    
                except Exception as e:
                    print(f"Model prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"ML model prediction failed: {str(e)}")
            else:
                raise Exception("Car loan ML model not available. Cannot process loan prediction.")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Car loan prediction failed: {str(e)}")