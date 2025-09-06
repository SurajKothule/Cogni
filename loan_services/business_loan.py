from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import pickle
import re
import json
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

IMPORTANT GUIDELINES TO AVOID REPETITIVE QUESTIONS:
1) ALWAYS check what information you already have before asking questions
2) NEVER ask for information that has already been provided and validated
3) Be conversational and friendly, like a helpful business loan specialist
4) Ask 1-2 related questions at a time to avoid overwhelming the customer
5) Provide brief explanations when needed (e.g., "Annual revenue is your total yearly business income")
6) If user provides partial info, acknowledge it positively and ask for missing details
7) Validate responses and ask for clarification if unclear
8) For categorical fields, provide the exact options to choose from
9) When you have ALL information, say exactly: INFORMATION_COMPLETE
10) Do NOT provide loan predictions - only collect information professionally
11) TRACK PROGRESS: Acknowledge what's been collected and what's still needed

CONVERSATION FLOW EXAMPLE:
- If you have name but need email: "Thanks [Name]! Now I need your email address for the application."
- If you have basic info but need business details: "Great! Now let's discuss your business. How many years has your business been operating?"
- Always acknowledge received information before asking for the next piece

Start by introducing yourself as a business loan specialist and ask for their name ONLY."""
    
    def get_fallback_greeting(self) -> str:
        return "Hello! I'm a business loan specialist here to help you with your business loan application. Business loans can help expand your operations, purchase equipment, or manage cash flow. Let's start with your full name - what should I call you?"
    
    def get_extraction_prompt(self, user_text: str, conversation: List[Dict[str, str]]) -> str:
        return f"""
Based on the conversation history and the user's latest response, extract any business loan-related information.

Conversation so far: {conversation[-5:] if len(conversation) > 5 else conversation}

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
- Be more flexible with name extraction - if user gives just first name when asked for name, extract it

Return ONLY a JSON object with the extracted fields. If no information is found, return empty JSON {{}}.
Example: {{"Customer_Name": "John Doe", "Business_Age_Years": 5, "Annual_Revenue": 2000000, "Net_Profit": 500000, "Business_Type": "Manufacturing", "Has_Collateral": "Yes"}}
""".strip()
    
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
                import re
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, str(value)):
                    return False, "Please provide a valid email address (e.g., name@company.com)."
                return True, ""
                
            elif field_name == "Customer_Phone":
                phone_str = str(value).replace(' ', '').replace('-', '').replace('(', '').replace(')', '').replace('+91', '')
                if not phone_str.isdigit() or len(phone_str) != 10 or phone_str[0] not in '6789':
                    return False, "Please provide a valid 10-digit mobile number starting with 6, 7, 8, or 9."
                return True, ""
                
            elif field_name == "Business_Age_Years":
                age = float(value)
                if age < 1:
                    return False, "INELIGIBLE: Business must be operating for at least 1 year to qualify for a business loan."
                elif age > 50:
                    return False, "Please verify your business age. The duration seems unusually high. Could you confirm how many years your business has been operating?"
                return True, ""
                    
            elif field_name == "CIBIL_Score":
                cibil = float(value)
                if cibil < 650:
                    return False, "INELIGIBLE: A minimum CIBIL score of 650 is required for business loan approval. Your current score does not meet our eligibility criteria."
                elif not (300 <= cibil <= 900):
                    return False, "Please provide a valid CIBIL score between 300 and 900. Could you check and confirm your credit score?"
                return True, ""
                    
            elif field_name == "Annual_Revenue":
                revenue = float(value)
                if revenue <= 0:
                    return False, "Annual revenue must be a positive amount. Please provide your yearly business revenue."
                elif revenue < 500000:  # Minimum 5 lakhs per year
                    return False, "INELIGIBLE: Minimum annual revenue of ₹5,00,000 is required for business loan eligibility."
                elif revenue > 1000000000:  # Maximum 100 crores (reasonable upper limit)
                    return False, "Please verify your annual revenue. The amount seems unusually high. Could you confirm your yearly business income?"
                return True, ""
                    
            elif field_name == "Net_Profit":
                profit = float(value)
                if profit <= 0:
                    return False, "Net profit must be a positive amount. Please provide your yearly net profit after all expenses."
                elif profit > 500000000:  # Maximum 50 crores (reasonable upper limit)
                    return False, "Please verify your net profit. The amount seems unusually high. Could you confirm your yearly net profit?"
                return True, ""
                    
            elif field_name == "Business_Type":
                valid_types = ["Retail", "Trading", "Services", "Manufacturing"]
                if value not in valid_types:
                    return False, f"Please select your business type from: {', '.join(valid_types)}. Which category best describes your business?"
                return True, ""
                    
            elif field_name == "Loan_Tenure_Years":
                tenure = float(value)
                if not (1 <= tenure <= 10):
                    return False, "Business loan tenure must be between 1 and 10 years. Please specify your preferred repayment period."
                return True, ""
                    
            elif field_name == "Existing_Loan_Amount":
                amount = float(value)
                if amount < 0:
                    return False, "Existing loan amount cannot be negative. Please provide your current business loan amount (enter 0 if none)."
                return True, ""
                    
            elif field_name == "Has_Collateral":
                if value not in ["Yes", "No"]:
                    return False, "Please specify if you have collateral available: Yes or No."
                return True, ""
                    
            elif field_name == "Has_Guarantor":
                if value not in ["Yes", "No"]:
                    return False, "Please specify if you have a guarantor available: Yes or No."
                return True, ""
                    
            elif field_name == "Industry_Risk_Rating":
                valid_industries = ["Healthcare", "FMCG", "IT Services", "Education", "Automobile", "Telecom", "Real Estate", "Hospitality", "Crypto", "Airlines"]
                if value not in valid_industries:
                    return False, f"Please select your industry from: {', '.join(valid_industries)}. Which industry best describes your business?"
                return True, ""
                    
            elif field_name == "Location_Tier":
                valid_locations = ["Tier-1 City", "Tier-2 City", "Tier-3 City", "Rural"]
                if value not in valid_locations:
                    return False, f"Please select your business location type from: {', '.join(valid_locations)}. Which category best describes your business location?"
                return True, ""
                    
            elif field_name == "Expected_Loan_Amount":
                amount = float(value)
                if amount <= 0:
                    return False, "Expected loan amount must be a positive amount. Please specify how much loan you need."
                elif amount < 100000:  # Minimum 1 lakh
                    return False, "INELIGIBLE: Minimum loan amount is ₹1,00,000 for business loans."
                elif amount > 100000000:  # Maximum 10 crores
                    return False, "Please verify your loan requirement. The amount seems unusually high. Could you confirm how much loan you need?"
                return True, ""
                    
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

    def convert_amount_to_number(self, amount_str: str) -> float:
        """Convert lakh/crore amounts to numbers with better error handling"""
        if isinstance(amount_str, (int, float)):
            return float(amount_str)
            
        amount_str = str(amount_str).lower().strip()
        
        # Remove currency symbols and extra spaces
        amount_str = re.sub(r'[₹rs\.\s]+', '', amount_str)
        
        # Extract number and unit - handle commas in numbers
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
        """Enhanced fallback extraction with better context awareness"""
        extracted = {}
        text_lower = user_text.lower().strip()
        
        # Get context from the last assistant message
        last_assistant_msg = ""
        if len(conversation) > 0:
            for msg in reversed(conversation):
                if msg.get('role') == 'assistant':
                    last_assistant_msg = msg.get('content', '').lower()
                    break
        
        print(f"DEBUG - Context: {last_assistant_msg[:100]}...")
        print(f"DEBUG - User input: '{user_text}'")
        
        # 1. CUSTOMER NAME - Enhanced patterns
        if not extracted.get('Customer_Name'):
            name_patterns = [
                r"(?:my name is|i am|i'm|call me|name:|this is)\s+([a-zA-Z\s]{2,30})",
                r"^([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*$",  # Simple name format
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, user_text, re.IGNORECASE)
                if match:
                    name = match.group(1).strip().title()
                    # Filter out common false positives
                    if not any(word in name.lower() for word in ['years', 'old', 'work', 'job', 'salary', 'business', 'company', 'manufacturing', 'retail', 'services', 'trading']):
                        extracted['Customer_Name'] = name
                        break
            
            # Context-aware name extraction when asked for name
            if not extracted.get('Customer_Name') and ('name' in last_assistant_msg or 'call you' in last_assistant_msg):
                words = user_text.strip().split()
                if 1 <= len(words) <= 3 and all(re.match(r'^[a-zA-Z]+$', word) for word in words):
                    if not any(word.lower() in ['yes', 'no', 'ok', 'sure', 'hello', 'hi', 'good', 'fine'] for word in words):
                        extracted['Customer_Name'] = user_text.strip().title()
        
        # 2. PHONE NUMBER - Enhanced extraction
        phone_patterns = [
            r'(?:\+?91[\s-]?)?([6-9]\d{9})',
            r'(?:phone|mobile|contact|number)[\s:]*(\+?91)?[\s-]*([6-9]\d{9})',
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, user_text)
            if match:
                phone = match.group(-1)  # Get last group (the actual number)
                if len(phone) == 10 and phone[0] in '6789':
                    extracted['Customer_Phone'] = phone
                    break
        
        # Context-aware phone extraction
        if not extracted.get('Customer_Phone') and any(word in last_assistant_msg for word in ['phone', 'mobile', 'contact', 'number']):
            phone_digits = re.sub(r'[^\d]', '', user_text)
            if len(phone_digits) == 10 and phone_digits[0] in '6789':
                extracted['Customer_Phone'] = phone_digits
        
        # 3. EMAIL - Enhanced extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, user_text)
        if email_match:
            extracted['Customer_Email'] = email_match.group(0)
        
        # 4. BUSINESS AGE - Enhanced patterns
        business_age_patterns = [
            r'(?:business|company|operating|established).*?(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?).*?(?:business|company|operating|established)',
            r'(?:since|for|past)\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:old|experience|in business)',
        ]
        
        for pattern in business_age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = int(match.group(1))
                if 1 <= age <= 50:
                    extracted['Business_Age_Years'] = age
                    break
        
        # Context-aware business age extraction
        if not extracted.get('Business_Age_Years') and any(word in last_assistant_msg for word in ['business', 'operating', 'years', 'how long', 'how many']):
            number_match = re.search(r'\b(\d{1,2})\b', user_text)
            if number_match:
                age = int(number_match.group(1))
                if 1 <= age <= 50:
                    extracted['Business_Age_Years'] = age
        
        # 5. REVENUE/PROFIT - Enhanced with better amount conversion
        revenue_patterns = [
            r'(?:revenue|turnover|income|earning).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'(?:make|earn|generate).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:revenue|turnover|yearly)',
        ]
        
        for pattern in revenue_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount >= 100000:  # Minimum reasonable revenue
                    extracted['Annual_Revenue'] = amount
                    break
        
        # Context-aware revenue extraction
        if not extracted.get('Annual_Revenue') and any(word in last_assistant_msg for word in ['revenue', 'turnover', 'income', 'yearly']):
            amount_match = re.search(r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)', text_lower)
            if amount_match:
                amount = self.convert_amount_to_number(amount_match.group(1))
                if amount >= 100000:
                    extracted['Annual_Revenue'] = amount
        
        # 6. NET PROFIT - Enhanced patterns
        profit_patterns = [
            r'(?:profit|net profit).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:profit|net)',
        ]
        
        for pattern in profit_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount >= 10000:  # Minimum reasonable profit
                    extracted['Net_Profit'] = amount
                    break
        
        # Context-aware profit extraction
        if not extracted.get('Net_Profit') and 'profit' in last_assistant_msg:
            amount_match = re.search(r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)', text_lower)
            if amount_match:
                amount = self.convert_amount_to_number(amount_match.group(1))
                if amount >= 10000:
                    extracted['Net_Profit'] = amount
        
        # 7. CIBIL SCORE - Enhanced patterns
        cibil_patterns = [
            r'(?:cibil|credit.*?score).*?(\d{3})',
            r'(\d{3}).*?(?:cibil|credit.*?score)',
            r'(?:score|rating).*?(\d{3})',
        ]
        
        for pattern in cibil_patterns:
            match = re.search(pattern, text_lower)
            if match:
                score = int(match.group(1))
                if 300 <= score <= 900:
                    extracted['CIBIL_Score'] = score
                    break
        
        # Context-aware CIBIL extraction
        if not extracted.get('CIBIL_Score') and any(word in last_assistant_msg for word in ['cibil', 'credit score', 'score']):
            score_match = re.search(r'\b(\d{3})\b', user_text)
            if score_match:
                score = int(score_match.group(1))
                if 300 <= score <= 900:
                    extracted['CIBIL_Score'] = score
        
        # 8. BUSINESS TYPE - Enhanced mapping
        business_type_keywords = {
            'retail': 'Retail', 'shop': 'Retail', 'store': 'Retail', 'selling': 'Retail',
            'trading': 'Trading', 'trade': 'Trading', 'import': 'Trading', 'export': 'Trading',
            'service': 'Services', 'services': 'Services', 'consulting': 'Services', 'agency': 'Services',
            'manufacturing': 'Manufacturing', 'manufacture': 'Manufacturing', 'factory': 'Manufacturing', 'production': 'Manufacturing'
        }
        
        for keyword, business_type in business_type_keywords.items():
            if keyword in text_lower:
                extracted['Business_Type'] = business_type
                break
        
        # 9. YES/NO FIELDS - Enhanced with better context awareness
        yes_indicators = ['yes', 'yeah', 'yep', 'have', 'available', 'got', 'do have', 'we have', 'there is', 'exists']
        no_indicators = ['no', "don't", "dont", "not", "none", "don't have", "do not have", "not available", "no we", "not have"]
        
        has_yes = any(indicator in text_lower for indicator in yes_indicators)
        has_no = any(indicator in text_lower for indicator in no_indicators)
        
        if has_yes and not has_no:
            if 'collateral' in last_assistant_msg:
                extracted['Has_Collateral'] = 'Yes'
            elif 'guarantor' in last_assistant_msg:
                extracted['Has_Guarantor'] = 'Yes'
        elif has_no and not has_yes:
            if 'collateral' in last_assistant_msg:
                extracted['Has_Collateral'] = 'No'
            elif 'guarantor' in last_assistant_msg:
                extracted['Has_Guarantor'] = 'No'
        
        # 10. LOAN AMOUNT - Enhanced patterns
        loan_amount_patterns = [
            r'(?:need|want|require|looking for).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'(?:loan.*?amount|amount.*?loan).*?([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)',
            r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?).*?(?:loan|amount)',
        ]
        
        for pattern in loan_amount_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = self.convert_amount_to_number(match.group(1))
                if amount >= 100000:  # Minimum 1 lakh
                    extracted['Expected_Loan_Amount'] = amount
                    break
        
        # Context-aware loan amount extraction
        if not extracted.get('Expected_Loan_Amount') and any(word in last_assistant_msg for word in ['loan amount', 'how much', 'amount need', 'amount require']):
            amount_match = re.search(r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)', text_lower)
            if amount_match:
                amount = self.convert_amount_to_number(amount_match.group(1))
                if amount >= 100000:
                    extracted['Expected_Loan_Amount'] = amount
        
        # 11. LOAN TENURE - Enhanced patterns
        tenure_patterns = [
            r'(?:tenure|repayment|period|term).*?(\d{1,2})',
            r'(\d{1,2})\s*(?:years?|year|yrs?|yr)(?:\s*(?:tenure|repayment|period|term))?',
            r'(?:for|over|in)\s*(\d{1,2})\s*(?:years?|yrs?)',
        ]
        
        for pattern in tenure_patterns:
            match = re.search(pattern, text_lower)
            if match:
                val = int(match.group(1))
                if 1 <= val <= 10:
                    extracted['Loan_Tenure_Years'] = val
                    break
        
        # Context-aware tenure extraction
        if not extracted.get('Loan_Tenure_Years') and any(word in last_assistant_msg for word in ['tenure', 'repayment', 'period', 'years']):
            number_match = re.search(r'(\d{1,2})', text_lower)
            if number_match:
                val = int(number_match.group(1))
                if 1 <= val <= 10:
                    extracted['Loan_Tenure_Years'] = val
        
        # 12. EXISTING LOAN AMOUNT - Enhanced patterns
        if any(word in text_lower for word in ['existing', 'current', 'outstanding', 'emi']):
            if any(word in text_lower for word in ['none', 'no', 'zero', '0', 'not have']):
                extracted['Existing_Loan_Amount'] = 0
            else:
                amount_match = re.search(r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)', text_lower)
                if amount_match:
                    amount = self.convert_amount_to_number(amount_match.group(1))
                    if amount >= 0:
                        extracted['Existing_Loan_Amount'] = amount
        
        # Context-aware existing loan extraction
        if not extracted.get('Existing_Loan_Amount') and 'existing' in last_assistant_msg:
            if any(word in text_lower for word in ['none', 'no', 'zero', '0']):
                extracted['Existing_Loan_Amount'] = 0
            else:
                amount_match = re.search(r'([\d,]+(?:\.[\d,]+)?\s*(?:lakh|lakhs|crore|crores)?)', text_lower)
                if amount_match:
                    amount = self.convert_amount_to_number(amount_match.group(1))
                    if amount >= 0:
                        extracted['Existing_Loan_Amount'] = amount
        
        # 13. INDUSTRY SELECTION - Enhanced mapping
        industry_keywords = {
            'healthcare': 'Healthcare', 'health': 'Healthcare', 'medical': 'Healthcare', 'hospital': 'Healthcare',
            'fmcg': 'FMCG', 'consumer goods': 'FMCG', 'goods': 'FMCG',
            'it services': 'IT Services', 'it service': 'IT Services', 'software': 'IT Services', 'tech': 'IT Services', 'technology': 'IT Services',
            'education': 'Education', 'school': 'Education', 'college': 'Education', 'training': 'Education',
            'automobile': 'Automobile', 'auto': 'Automobile', 'car': 'Automobile', 'vehicle': 'Automobile',
            'telecom': 'Telecom', 'telecommunications': 'Telecom', 'mobile': 'Telecom', 'network': 'Telecom',
            'real estate': 'Real Estate', 'property': 'Real Estate', 'construction': 'Real Estate', 'builder': 'Real Estate',
            'hospitality': 'Hospitality', 'hotel': 'Hospitality', 'restaurant': 'Hospitality', 'food': 'Hospitality', 'tourism': 'Hospitality',
            'crypto': 'Crypto', 'cryptocurrency': 'Crypto', 'blockchain': 'Crypto',
            'airline': 'Airlines', 'airlines': 'Airlines', 'aviation': 'Airlines', 'flight': 'Airlines'
        }
        
        for keyword, industry in industry_keywords.items():
            if keyword in text_lower:
                extracted['Industry_Risk_Rating'] = industry
                break
        
        # 14. LOCATION TIER - Enhanced patterns
        location_keywords = {
            'tier-1': 'Tier-1 City', 'tier 1': 'Tier-1 City', 'metro': 'Tier-1 City', 'metropolitan': 'Tier-1 City',
            'mumbai': 'Tier-1 City', 'delhi': 'Tier-1 City', 'bangalore': 'Tier-1 City', 'chennai': 'Tier-1 City', 'kolkata': 'Tier-1 City', 'hyderabad': 'Tier-1 City', 'pune': 'Tier-1 City',
            'tier-2': 'Tier-2 City', 'tier 2': 'Tier-2 City',
            'tier-3': 'Tier-3 City', 'tier 3': 'Tier-3 City', 'small city': 'Tier-3 City',
            'rural': 'Rural', 'village': 'Rural', 'town': 'Rural'
        }
        
        for keyword, location in location_keywords.items():
            if keyword in text_lower:
                extracted['Location_Tier'] = location
                break
        
        # Context-aware location extraction
        if not extracted.get('Location_Tier') and any(word in last_assistant_msg for word in ['location', 'tier', 'city', 'where']):
            tier_match = re.search(r'tier\s*[- ]?(1|2|3)', text_lower)
            if tier_match:
                num = tier_match.group(1)
                extracted['Location_Tier'] = f'Tier-{num} City'
            elif re.search(r'\b1\b', text_lower):
                extracted['Location_Tier'] = 'Tier-1 City'
            elif re.search(r'\b2\b', text_lower):
                extracted['Location_Tier'] = 'Tier-2 City'
            elif re.search(r'\b3\b', text_lower):
                extracted['Location_Tier'] = 'Tier-3 City'
        
        print(f"DEBUG - Final extracted information: {extracted}")
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