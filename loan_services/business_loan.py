from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import pickle
from .base_loan import BaseLoanService

class BusinessLoanService(BaseLoanService):
    """Business Loan Service with ML Model Integration"""
    
    def get_required_fields(self) -> List[str]:
        return [
            # Customer Contact Information
            "Customer_Name",
            "Customer_Email", 
            "Customer_Phone",
            # Model Prediction Fields
            "Business_Age_Years",
            "Annual_Revenue",
            "Net_Profit",
            "CIBIL_Score",
            "Business_Type",
            "Existing_Loan_Amount",
            "Loan_Tenure_Years",
            "Has_Collateral",
            "Has_Guarantor",
            "Industry_Risk_Rating",
            "Location_Tier",
            "Expected_Loan_Amount",
        ]
    
    def get_model_files(self) -> Dict[str, str]:
        return {
            "business_loan_model": "business_loan_model.pkl",
        }
    
    def get_system_prompt(self) -> str:
        return """You are a friendly and professional business loan advisor chatbot.

Your task is to systematically collect the following information from users through natural conversation:

Required Customer Information:
- Customer_Name (full name)
- Customer_Email (email address)
- Customer_Phone (10-digit phone number)

Required Business Information:
- Business_Age_Years (how many years the business has been operating)
- Annual_Revenue (yearly business revenue in INR)
- Net_Profit (yearly net profit in INR)
- CIBIL_Score (credit score, 300-900, minimum 650 for business loans)
- Business_Type: exactly one of ["Retail", "Trading", "Services", "Manufacturing"]
- Existing_Loan_Amount (current business loan amount in INR, can be 0)
- Loan_Tenure_Years (repayment period in years, typically 1-10)
- Has_Collateral: "Yes" or "No" (whether business has collateral to offer)
- Has_Guarantor: "Yes" or "No" (whether business has a guarantor)
- Industry_Risk_Rating: Ask user to select their industry from: Healthcare, FMCG, IT Services, Education, Automobile, Telecom, Real Estate, Hospitality, Crypto, Airlines
- Location_Tier: Ask user about their business location type: Tier-1 City, Tier-2 City, Tier-3 City, Rural
- Expected_Loan_Amount: How much loan amount they need in INR (must be positive)

Guidelines:
1) Be conversational and friendly, like a helpful business loan specialist.
2) Ask 1-2 related questions at a time to avoid overwhelming the customer.
3) Provide brief explanations when needed (e.g., "Annual revenue is your total yearly business income").
4) If user provides partial info, acknowledge it positively and ask for missing details.
5) Validate responses and ask for clarification if unclear.
6) For categorical fields, provide the exact options to choose from.
7) When you have ALL information, say exactly: INFORMATION_COMPLETE
8) Do NOT provide loan predictions - only collect information professionally.

Start by introducing yourself as a business loan specialist."""
    
    def get_fallback_greeting(self) -> str:
        return "Hello! I'm a business loan specialist here to help you with your business loan application. Business loans can help expand your operations, purchase equipment, or manage cash flow. Let's start with your full name - what should I call you?"
    
    def get_extraction_prompt(self, user_text: str, conversation: List[Dict[str, str]]) -> str:
        return f"""
Based on the conversation history and the user's latest response, extract any business loan-related information.

Conversation so far: {conversation[-3:] if len(conversation) > 3 else conversation}

User's latest response: "{user_text}"

Extract information for these fields (only if clearly mentioned):

Customer Information:
- Customer_Name: full name as string
- Customer_Email: email address as string  
- Customer_Phone: 10-digit phone number as string (remove +91, spaces, dashes)

Business Information:
- Business_Age_Years: number (years business has been operating)
- Annual_Revenue: number in INR (yearly business revenue, must be positive)
- Net_Profit: number in INR (yearly net profit, must be positive)
- CIBIL_Score: number (300-900, minimum 650 for business loans)
- Business_Type: exactly one of ["Retail", "Trading", "Services", "Manufacturing"]
- Existing_Loan_Amount: number in INR (current business loan amount, 0 if none)
- Loan_Tenure_Years: number (years, typically 1-10)
- Has_Collateral: "Yes" or "No" (whether business has collateral)
- Has_Guarantor: "Yes" or "No" (whether business has guarantor)
- Industry_Risk_Rating: map user's industry to one of ["Healthcare", "FMCG", "IT Services", "Education", "Automobile", "Telecom", "Real Estate", "Hospitality", "Crypto", "Airlines"]
- Location_Tier: map user's location to one of ["Tier-1 City", "Tier-2 City", "Tier-3 City", "Rural"]
- Expected_Loan_Amount: number in INR (loan amount they need, must be positive)

Important conversion rules:
- Convert lakhs/crores to actual numbers: "20 lakh" = 2000000, "5 lakh" = 500000, "1.5 crore" = 15000000
- For Business_Type, map variations like "retail business", "manufacturing company" to exact options
- For Has_Collateral/Has_Guarantor, map "yes", "have", "available" to "Yes" and "no", "don't have" to "No"
- Extract only information that is clearly stated

Return ONLY a JSON object with the extracted fields. If no information is found, return empty JSON {{}}.
Example: {{"Customer_Name": "John Doe", "Business_Age_Years": 5, "Annual_Revenue": 2000000, "Net_Profit": 500000, "Business_Type": "Manufacturing", "Has_Collateral": "Yes"}}
""".strip()
    
    def validate_field(self, field_name: str, value: Any) -> Tuple[bool, str]:
        """Validate individual field values with strict eligibility criteria"""
        try:
            if field_name == "Business_Age_Years":
                age = float(value)
                if age < 1:
                    return False, "INELIGIBLE: Business must be operating for at least 1 year to qualify for a business loan."
                elif age > 50:
                    return False, "Please verify your business age. The duration seems unusually high. Could you confirm how many years your business has been operating?"
                    
            elif field_name == "CIBIL_Score":
                cibil = float(value)
                if cibil < 650:
                    return False, "INELIGIBLE: A minimum CIBIL score of 650 is required for business loan approval. Your current score does not meet our eligibility criteria."
                elif not (300 <= cibil <= 900):
                    return False, "Please provide a valid CIBIL score between 300 and 900. Could you check and confirm your credit score?"
                    
            elif field_name == "Annual_Revenue":
                revenue = float(value)
                if revenue <= 0:
                    return False, "Annual revenue must be a positive amount. Please provide your yearly business revenue."
                elif revenue < 500000:  # Minimum 5 lakhs per year
                    return False, "INELIGIBLE: Minimum annual revenue of ₹5,00,000 is required for business loan eligibility."
                elif revenue > 1000000000:  # Maximum 100 crores (reasonable upper limit)
                    return False, "Please verify your annual revenue. The amount seems unusually high. Could you confirm your yearly business income?"
                    
            elif field_name == "Net_Profit":
                profit = float(value)
                if profit <= 0:
                    return False, "Net profit must be a positive amount. Please provide your yearly net profit after all expenses."
                elif profit > 500000000:  # Maximum 50 crores (reasonable upper limit)
                    return False, "Please verify your net profit. The amount seems unusually high. Could you confirm your yearly net profit?"
                    
            elif field_name == "Business_Type":
                valid_types = ["Retail", "Trading", "Services", "Manufacturing"]
                if value not in valid_types:
                    return False, f"Please select your business type from: {', '.join(valid_types)}. Which category best describes your business?"
                    
            elif field_name == "Loan_Tenure_Years":
                tenure = float(value)
                if not (1 <= tenure <= 10):
                    return False, "Business loan tenure must be between 1 and 10 years. Please specify your preferred repayment period."
                    
            elif field_name == "Existing_Loan_Amount":
                amount = float(value)
                if amount < 0:
                    return False, "Existing loan amount cannot be negative. Please provide your current business loan amount (enter 0 if none)."
                    
            elif field_name == "Has_Collateral":
                if value not in ["Yes", "No"]:
                    return False, "Please specify if you have collateral available: Yes or No."
                    
            elif field_name == "Has_Guarantor":
                if value not in ["Yes", "No"]:
                    return False, "Please specify if you have a guarantor available: Yes or No."
                    
            elif field_name == "Industry_Risk_Rating":
                valid_industries = ["Healthcare", "FMCG", "IT Services", "Education", "Automobile", "Telecom", "Real Estate", "Hospitality", "Crypto", "Airlines"]
                if value not in valid_industries:
                    return False, f"Please select your industry from: {', '.join(valid_industries)}. Which industry best describes your business?"
                    
            elif field_name == "Location_Tier":
                valid_locations = ["Tier-3 City", "Tier-1 City", "Tier-2 City", "Rural"]
                if value not in valid_locations:
                    return False, f"Please select your business location type from: {', '.join(valid_locations)}. Which category best describes your business location?"
                    
            elif field_name == "Customer_Phone":
                # Remove any spaces, dashes, or other characters
                phone_clean = str(value).replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace("+91", "")
                if not phone_clean.isdigit() or len(phone_clean) != 10:
                    return False, "Please provide a valid 10-digit phone number (e.g., 9876543210)."
                    
            elif field_name == "Expected_Loan_Amount":
                amount = float(value)
                if amount <= 0:
                    return False, "Expected loan amount must be a positive amount. Please specify how much loan you need."
                elif amount < 100000:  # Minimum 1 lakh
                    return False, "INELIGIBLE: Minimum loan amount is ₹1,00,000 for business loans."
                elif amount > 100000000:  # Maximum 10 crores
                    return False, "Please verify your loan requirement. The amount seems unusually high. Could you confirm how much loan you need?"
                    
            return True, ""
            
        except (ValueError, TypeError):
            field_display = field_name.replace('_', ' ').lower()
            return False, f"Please provide a valid {field_display} in the correct format."
    
    def validate_business_logic(self, collected_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate business logic rules across multiple fields"""
        # Check if net profit is less than annual revenue
        if 'Net_Profit' in collected_info and 'Annual_Revenue' in collected_info:
            net_profit = float(collected_info['Net_Profit'])
            annual_revenue = float(collected_info['Annual_Revenue'])
            
            if net_profit >= annual_revenue:
                return False, "VALIDATION ERROR: Net profit cannot be equal to or greater than annual revenue. Please verify your financial figures. Net profit should be the amount left after all business expenses are deducted from revenue."
        
        return True, ""
    
    def convert_location_tier_to_numeric(self, location_tier: str) -> int:
        """Convert location tier to numeric value for ML model"""
        location_tier_map = {
            "Tier-1 City": 1,
            "Tier-2 City": 2, 
            "Tier-3 City": 3,
            "Rural": 4
        }
        return location_tier_map.get(location_tier, 3)  # Default to Tier-2 City if unknown

    def extract_info_from_response(self, user_text: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract information from user response with business-specific fallback logic"""
        # Try OpenAI first
        if self.client:
            extraction_prompt = self.get_extraction_prompt(user_text, conversation)
            
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0
                )
                extracted_text = resp.choices[0].message.content.strip()
                import re, json
                m = re.search(r"\{.*\}", extracted_text, re.DOTALL)
                if m:
                    return json.loads(m.group())
            except Exception as e:
                print(f"OpenAI extraction failed: {e}")
        
        # Fallback to business-specific pattern matching
        return self._business_fallback_extraction(user_text, conversation)
    
    def _business_fallback_extraction(self, user_text: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Business-specific fallback extraction with lakh/crore conversion"""
        extracted = {}
        text_lower = user_text.lower().strip()
        
        import re
        
        # Extract name patterns
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
        
        # If no pattern matched but it looks like just a name (when asked for name)
        if not extracted.get('Customer_Name'):
            # Check if previous message was asking for name
            if len(conversation) > 0:
                last_assistant_msg = ""
                for msg in reversed(conversation):
                    if msg.get('role') == 'assistant':
                        last_assistant_msg = msg.get('content', '').lower()
                        break
                
                if 'name' in last_assistant_msg and len(text_lower.split()) <= 3:
                    # Simple name check - only letters and spaces, reasonable length
                    if re.match(r'^[a-zA-Z\s]+$', user_text.strip()) and 2 <= len(user_text.strip()) <= 50:
                        extracted['Customer_Name'] = user_text.strip().title()
        
        # Extract phone numbers
        phone_pattern = r'(\+?91[-\s]?)?([6-9]\d{9})'
        phone_match = re.search(phone_pattern, user_text)
        if phone_match:
            extracted['Customer_Phone'] = phone_match.group(2)
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, user_text)
        if email_match:
            extracted['Customer_Email'] = email_match.group(0)
        
        # Extract business age
        business_age_patterns = [
            r'business.*?(\d+)\s+years?',
            r'operating.*?(\d+)\s+years?',
            r'(\d+)\s+years?.*?business',
            r'(\d+)\s+years?.*?operating'
        ]
        
        for pattern in business_age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = int(match.group(1))
                if 1 <= age <= 50:
                    extracted['Business_Age_Years'] = age
                    break
        
        # Extract revenue/profit with lakh/crore conversion
        def convert_amount(amount_str):
            """Convert lakh/crore amounts to numbers"""
            amount_str = amount_str.lower().strip()
            
            # Extract number and unit - handle commas in numbers
            number_match = re.search(r'([\d,]+(?:\.[\d,]+)?)', amount_str)
            if not number_match:
                return None
            
            # Remove commas and convert to float
            number_str = number_match.group(1).replace(',', '')
            try:
                number = float(number_str)
            except ValueError:
                return None
            
            if 'crore' in amount_str:
                return int(number * 10000000)  # 1 crore = 1,00,00,000
            elif 'lakh' in amount_str:
                return int(number * 100000)    # 1 lakh = 1,00,000
            else:
                return int(number)
        
        # Check what the last assistant message was asking for
        last_assistant_msg = ""
        if len(conversation) > 0:
            for msg in reversed(conversation):
                if msg.get('role') == 'assistant':
                    last_assistant_msg = msg.get('content', '').lower()
                    break
        
        # Extract revenue patterns - handle comma-separated numbers
        revenue_patterns = [
            r'revenue.*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?)',
            r'turnover.*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?).*?revenue',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?).*?turnover'
        ]
        
        for pattern in revenue_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = convert_amount(match.group(1))
                if amount and amount >= 100000:  # Minimum reasonable revenue
                    extracted['Annual_Revenue'] = amount
                    break
        
        # If asking about revenue and user just gives amount
        if 'revenue' in last_assistant_msg and not extracted.get('Annual_Revenue'):
            amount_match = re.search(r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?)', text_lower)
            if amount_match:
                amount = convert_amount(amount_match.group(1))
                if amount and amount >= 100000:
                    extracted['Annual_Revenue'] = amount
        
        # Extract profit patterns - handle comma-separated numbers
        profit_patterns = [
            r'profit.*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?).*?profit'
        ]
        
        for pattern in profit_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = convert_amount(match.group(1))
                if amount and amount >= 10000:  # Minimum reasonable profit
                    extracted['Net_Profit'] = amount
                    break
        
        # If asking about profit and user just gives amount
        if 'profit' in last_assistant_msg and not extracted.get('Net_Profit'):
            amount_match = re.search(r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?)', text_lower)
            if amount_match:
                amount = convert_amount(amount_match.group(1))
                if amount and amount >= 10000:
                    extracted['Net_Profit'] = amount
        
        # Extract CIBIL score
        cibil_patterns = [
            r'cibil.*?(\d{3})',
            r'credit.*?score.*?(\d{3})',
            r'(\d{3}).*?cibil',
            r'(\d{3}).*?credit.*?score'
        ]
        
        for pattern in cibil_patterns:
            match = re.search(pattern, text_lower)
            if match:
                score = int(match.group(1))
                if 300 <= score <= 900:
                    extracted['CIBIL_Score'] = score
                    break
        
        # Extract business type
        business_type_map = {
            'retail': 'Retail',
            'trading': 'Trading', 
            'service': 'Services',
            'services': 'Services',
            'manufacturing': 'Manufacturing',
            'manufacture': 'Manufacturing'
        }
        
        for key, value in business_type_map.items():
            if key in text_lower:
                extracted['Business_Type'] = value
                break
        
        # Extract Yes/No fields
        if any(word in text_lower for word in ['yes', 'have', 'available', 'got']):
            if 'collateral' in last_assistant_msg:
                extracted['Has_Collateral'] = 'Yes'
            elif 'guarantor' in last_assistant_msg:
                extracted['Has_Guarantor'] = 'Yes'
        elif any(word in text_lower for word in ['no', "don't", "dont", "not", "none"]):
            if 'collateral' in last_assistant_msg:
                extracted['Has_Collateral'] = 'No'
            elif 'guarantor' in last_assistant_msg:
                extracted['Has_Guarantor'] = 'No'
        
        # Extract expected loan amount
        loan_amount_patterns = [
            r'need.*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?)',
            r'want.*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?)',
            r'require.*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?)',
            r'loan.*?amount.*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?).*?loan'
        ]
        
        for pattern in loan_amount_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = convert_amount(match.group(1))
                if amount and amount >= 100000:  # Minimum 1 lakh
                    extracted['Expected_Loan_Amount'] = amount
                    break
        
        # If asking about loan amount and user just gives amount
        if any(word in last_assistant_msg for word in ['loan amount', 'how much', 'amount need', 'amount require']):
            if not extracted.get('Expected_Loan_Amount'):
                amount_match = re.search(r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|crore|lakhs|crores)?)', text_lower)
                if amount_match:
                    amount = convert_amount(amount_match.group(1))
                    if amount and amount >= 100000:
                        extracted['Expected_Loan_Amount'] = amount
        
        return extracted

    def prepare_model_input(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data for the business loan model"""
        try:
            # Map categorical values to model format
            industry_risk_map = {
                "Healthcare": 1, "FMCG": 1, "IT Services": 2, "Education": 2,
                "Automobile": 3, "Telecom": 3, "Real Estate": 4, "Hospitality": 4,
                "Crypto": 5, "Airlines": 5
            }
            
            location_tier_map = {
                "Metro": 1, "Tier-1 City": 2, "Tier-2 City": 3, "Rural": 4
            }
            
            # Convert Yes/No to 1/0
            has_collateral = 1 if user_input['Has_Collateral'] == 'Yes' else 0
            has_guarantor = 1 if user_input['Has_Guarantor'] == 'Yes' else 0
            
            # Calculate derived features
            annual_revenue = float(user_input['Annual_Revenue'])
            net_profit = float(user_input['Net_Profit'])
            existing_loan = float(user_input.get('Existing_Loan_Amount', 0))
            
            profit_margin = (net_profit / annual_revenue) * 100
            debt_to_revenue_ratio = (existing_loan / annual_revenue) * 100
            
            # Create input dataframe matching your model's expected features
            input_data = {
                'Business_Age_Years': float(user_input['Business_Age_Years']),
                'Annual_Revenue': annual_revenue,
                'Net_Profit': net_profit,
                'CIBIL_Score': float(user_input['CIBIL_Score']),
                'Business_Type': user_input['Business_Type'],  # Will be encoded later
                'Existing_Loan_Amount': existing_loan,
                'Loan_Tenure_Years': float(user_input['Loan_Tenure_Years']),
                'Has_Collateral': has_collateral,
                'Has_Guarantor': has_guarantor,
                'Industry_Risk_Rating': industry_risk_map[user_input['Industry_Risk_Rating']],
                'Location_Tier': location_tier_map[user_input['Location_Tier']],
                'Profit_Margin': profit_margin,
                'Debt_to_Revenue_Ratio': debt_to_revenue_ratio
            }
            
            print(f"Business Loan Input data prepared: {input_data}")
            
            input_df = pd.DataFrame([input_data])
            
            print(f"Prepared input dataframe shape: {input_df.shape}")
            print(f"Input columns: {list(input_df.columns)}")
            
            return input_df
            
        except Exception as e:
            print(f"Error in prepare_model_input: {e}")
            raise e
    
    def predict_loan(self, user_input: Dict[str, Any]) -> tuple:
        """Predict business loan amount and interest rate using ML model"""
        try:
            print(f"Business Loan Prediction - Input: {user_input}")
            
            # Prepare input data
            input_df = self.prepare_model_input(user_input)
            print(f"Prepared input shape: {input_df.shape}")
            print(f"Input columns: {list(input_df.columns)}")
            
            # Try to use actual ML model if available
            if self.models.get("business_loan_model"):
                try:
                    print("Using ML model for prediction...")
                    
                    # Load the model package - matching your structure exactly
                    package = self.models["business_loan_model"]
                    model = package["model"]
                    business_type_encoder = package["business_type_encoder"]
                    feature_columns = package["feature_columns"]
                    target_columns = package["target_columns"]
                    
                    print("Model components loaded successfully")
                    print(f"Expected features: {feature_columns}")
                    print(f"Target variables: {target_columns}")
                    
                    # Prepare data for prediction
                    df_input = input_df.copy()
                    
                    # Encode Business_Type using your label encoder
                    df_input["Business_Type_encoded"] = business_type_encoder.transform(df_input["Business_Type"])
                    df_input = df_input.drop("Business_Type", axis=1)
                    print(f"After encoding Business_Type: {df_input['Business_Type_encoded'].values}")
                    
                    # Add engineered features as per your model
                    df_input['Revenue_to_Profit_Ratio'] = df_input['Annual_Revenue'] / (df_input['Net_Profit'] + 1)
                    df_input['Age_Revenue_Interaction'] = df_input['Business_Age_Years'] * np.log1p(df_input['Annual_Revenue'])
                    df_input['CIBIL_Revenue_Score'] = df_input['CIBIL_Score'] * np.log1p(df_input['Annual_Revenue']) / 1000000
                    df_input['Risk_Adjusted_Revenue'] = df_input['Annual_Revenue'] / (df_input['Industry_Risk_Rating'] + df_input['Location_Tier'])
                    df_input['Collateral_Guarantor_Score'] = df_input['Has_Collateral'] * 2 + df_input['Has_Guarantor']
                    df_input['Business_Stability_Score'] = (df_input['Business_Age_Years'] / 25) + ((df_input['CIBIL_Score'] - 600) / 300)
                    df_input['Debt_Service_Coverage'] = df_input['Net_Profit'] / (df_input['Existing_Loan_Amount'] * 0.12 + 1)
                    df_input['Location_Risk_Combined'] = df_input['Location_Tier'] + df_input['Industry_Risk_Rating']
                    
                    # Select features in the correct order
                    df_input = df_input[feature_columns]
                    print(f"After feature selection: {list(df_input.columns)}")
                    
                    # Make predictions
                    prediction = model.predict(df_input)[0]
                    print(f"Raw predictions: {prediction}")
                    
                    # Handle predictions as per your structure
                    # prediction[0] = Max_Loan_Amount_Offered
                    # prediction[1] = Interest_Rate
                    max_loan_amount = float(prediction[0])
                    interest_rate = float(prediction[1])
                    
                    print(f"ML Prediction - Loan: Rs.{max_loan_amount:,.0f}, Rate: {interest_rate:.2f}%")
                    
                    # Ensure reasonable bounds for business loans
                    max_loan_amount = max(max_loan_amount, 100000)    # Min 1 lakh
                    max_loan_amount = min(max_loan_amount, 100000000) # Max 10 crores
                    interest_rate = max(8.0, min(24.0, interest_rate)) # Between 8% and 24%
                    
                    print(f"Final ML Prediction - Loan: Rs.{max_loan_amount:,.0f}, Rate: {interest_rate:.2f}%")
                    return round(float(max_loan_amount), 0), round(float(interest_rate), 2)
                    
                except Exception as e:
                    print(f"Model prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"ML model prediction failed: {str(e)}")
            else:
                raise Exception("Business loan ML model not available. Cannot process loan prediction.")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Business loan prediction failed: {str(e)}")