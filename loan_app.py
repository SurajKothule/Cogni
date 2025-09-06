import os
import time
import uuid
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from loan_services.loan_factory import LoanServiceFactory
from customer_data.storage_manager import CustomerDataManager
from customer_data.mongodb_storage_manager import MongoDBStorageManager

# Load environment variables
load_dotenv()

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize storage managers
try:
    # Try MongoDB first
    mongodb_storage = MongoDBStorageManager()
    storage_manager = mongodb_storage
    print("✅ Using MongoDB for data storage")
except Exception as e:
    # Fallback to local storage
    storage_manager = CustomerDataManager()
    print(f"⚠️  MongoDB connection failed, using local storage: {e}")
    print("📁 Using local file storage as fallback")

# ---------- FastAPI app ----------
app = FastAPI(title="Multi-Loan Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- In-memory session store ----------
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---------- Schemas ----------
class StartChatRequest(BaseModel):
    loan_type: str = Field(..., description="Type of loan: education, home, personal, business, gold, or car")

class StartChatResponse(BaseModel):
    session_id: str
    loan_type: str
    message: str
    required_fields: List[str]

class MessageRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier returned by /chat/start")
    message: str = Field(..., description="User message")

class MessageResponse(BaseModel):
    message: str
    recorded: Dict[str, Any] = {}
    missing_fields: List[str] = []
    prediction: Optional[Dict[str, Any]] = None

class LoanTypesResponse(BaseModel):
    available_types: List[str]
    descriptions: Dict[str, str]

# ---------- Helper Functions ----------
def init_session(loan_type: str) -> str:
    """Initialize a new chat session"""
    service = LoanServiceFactory.get_service(loan_type, OPENAI_API_KEY)
    
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = {
        "loan_type": loan_type,
        "conversation": [{"role": "system", "content": service.get_system_prompt()}],
        "user_profile": {},
        "created_at": time.time(),
    }
    return session_id

def _to_float(v):
    """Convert various string formats to float with enhanced error handling"""
    if isinstance(v, (int, float)):
        return float(v)
    
    if v is None:
        return 0.0
        
    s = str(v).replace(",", "").strip().lower()
    
    # Handle empty strings
    if not s:
        return 0.0
    
    # Handle Indian number formats
    if s.endswith("l") or "lac" in s or "lakh" in s:
        import re
        num = re.sub(r"[^\d.]", "", s)
        try:
            return float(num) * 100000
        except ValueError:
            return 0.0
            
    if s.endswith("cr") or "crore" in s:
        import re
        num = re.sub(r"[^\d.]", "", s)
        try:
            return float(num) * 10000000
        except ValueError:
            return 0.0
    
    # Normal float conversion with error handling
    import re
    num_match = re.search(r"[\d.]+", s)
    if num_match:
        try:
            return float(num_match.group())
        except ValueError:
            return 0.0
    
    return 0.0

def clean_user_profile(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and validate user profile data"""
    cleaned = {}
    for k, v in user_profile.items():
        if v is not None and str(v).strip() != "":
            cleaned[k] = v
    return cleaned

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

@app.get("/loan-types", response_model=LoanTypesResponse)
def get_loan_types():
    """Get available loan types and their descriptions"""
    descriptions = {
        "education": "Loans for higher education, courses, and academic expenses",
        "home": "Loans for purchasing, constructing, or renovating residential properties", 
        "personal": "Unsecured loans for personal expenses like medical, travel, wedding, etc.",
        "business": "Loans for business expansion, working capital, and commercial purposes",
        "gold": "Secured loans against gold jewelry and ornaments",
        "car": "Loans for purchasing new and used cars with flexible repayment options"
    }
    
    return LoanTypesResponse(
        available_types=LoanServiceFactory.get_available_loan_types(),
        descriptions=descriptions
    )

@app.post("/chat/start", response_model=StartChatResponse)
def chat_start(request: StartChatRequest):
    """Start a new chat session for a specific loan type"""
    loan_type = request.loan_type.lower()
    
    if loan_type not in LoanServiceFactory.get_available_loan_types():
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid loan type. Available types: {LoanServiceFactory.get_available_loan_types()}"
        )
    
    try:
        service = LoanServiceFactory.get_service(loan_type, OPENAI_API_KEY)
        session_id = init_session(loan_type)
        
        conv = SESSIONS[session_id]["conversation"]
        greeting = service.assistant_greeting(conv)
        conv.append({"role": "assistant", "content": greeting})
        
        return StartChatResponse(
            session_id=session_id,
            loan_type=loan_type,
            message=greeting,
            required_fields=service.get_required_fields()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting chat: {str(e)}")

@app.post("/chat/message", response_model=MessageResponse)
def chat_message(req: MessageRequest):
    """Send a message in an existing chat session with enhanced processing"""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Invalid session_id.")

    state = SESSIONS[req.session_id]
    loan_type = state["loan_type"]
    conversation = state["conversation"]
    user_profile = state["user_profile"]
    
    try:
        service = LoanServiceFactory.get_service(loan_type, OPENAI_API_KEY)
        required_fields = service.get_required_fields()

        print(f"DEBUG - Starting message processing for {loan_type}")
        print(f"DEBUG - Required fields: {required_fields}")
        print(f"DEBUG - Current user profile keys: {list(user_profile.keys())}")
        print(f"DEBUG - User message: '{req.message}'")

        # Prefill user_profile from previous saved application if exists
        try:
            saved = None
            if hasattr(storage_manager, 'get_application_by_session'):
                saved = storage_manager.get_application_by_session(loan_type, req.session_id)
            if saved and isinstance(saved, dict):
                loan_data = saved.get('loan_data') or saved.get('prediction_result', {}).get('profile') or {}
                for k, v in loan_data.items():
                    if k in required_fields and k not in user_profile and v is not None:
                        user_profile[k] = v
                        print(f"DEBUG - Restored from storage: {k} = {v}")
                
                # Also hydrate customer info
                ci = saved.get('customer_info', {})
                if ci:
                    if ci.get('name') and 'Customer_Name' in required_fields and 'Customer_Name' not in user_profile:
                        user_profile['Customer_Name'] = ci['name']
                        print(f"DEBUG - Restored customer name: {ci['name']}")
                    if ci.get('email') and 'Customer_Email' in required_fields and 'Customer_Email' not in user_profile:
                        user_profile['Customer_Email'] = ci['email']
                        print(f"DEBUG - Restored customer email: {ci['email']}")
                    if ci.get('phone') and 'Customer_Phone' in required_fields and 'Customer_Phone' not in user_profile:
                        user_profile['Customer_Phone'] = ci['phone']
                        print(f"DEBUG - Restored customer phone: {ci['phone']}")
        except Exception as e:
            print(f"DEBUG - Storage retrieval failed (non-fatal): {e}")
        
        # Append user message to conversation
        conversation.append({"role": "user", "content": req.message})
        print(f"DEBUG - Added user message to conversation")

        # Extract fields from user response with enhanced error handling
        try:
            extracted = service.extract_info_from_response(req.message, conversation)
            print(f"DEBUG - Extracted fields: {extracted}")
        except Exception as e:
            print(f"DEBUG - Extraction failed: {e}")
            extracted = {}
        
        # Process extracted information with enhanced validation
        recorded_now = {}
        validation_errors = []
        processing_errors = []
        
        for k, v in extracted.items():
            print(f"DEBUG - Processing field: {k} = {v}")
            
            # Check if field is required and has a non-empty value
            if k in required_fields and v is not None and str(v).strip() != "":
                try:
                    # Validate the field if the service has validation method
                    if hasattr(service, 'validate_field'):
                        is_valid, error_msg = service.validate_field(k, v)
                        print(f"DEBUG - Validation for {k}: valid={is_valid}, error='{error_msg}'")
                        
                        if not is_valid:
                            validation_errors.append(error_msg)
                            print(f"DEBUG - Validation failed for {k}: {error_msg}")
                            continue
                        else:
                            print(f"DEBUG - Validation passed for {k}")
                    
                    # Handle special cases for different loan types
                    if k == "Academic_Score" and hasattr(service, 'convert_academic_score_to_performance'):
                        # Education loan special handling
                        user_profile[k] = v
                        user_profile["Academic_Performance"] = service.convert_academic_score_to_performance(float(v))
                        recorded_now[k] = v
                        recorded_now["Academic_Performance"] = user_profile["Academic_Performance"]
                        print(f"DEBUG - Stored {k}={v} and derived Academic_Performance={user_profile['Academic_Performance']}")
                    else:
                        # Regular field storage
                        user_profile[k] = v
                        recorded_now[k] = v
                        print(f"DEBUG - Stored {k} = {v}")
                        
                except Exception as field_error:
                    processing_errors.append(f"Error processing {k}: {str(field_error)}")
                    print(f"DEBUG - Processing error for {k}: {field_error}")
                    continue
            else:
                if k in required_fields:
                    print(f"DEBUG - Field {k} skipped: value is None or empty")
        
        # Check business logic validation for business loans
        if loan_type == "business" and hasattr(service, 'validate_business_logic'):
            try:
                is_valid, error_msg = service.validate_business_logic(user_profile)
                if not is_valid:
                    validation_errors.append(error_msg)
                    print(f"DEBUG - Business logic validation failed: {error_msg}")
            except Exception as e:
                print(f"DEBUG - Business logic validation error: {e}")
        
        # Clean user profile to remove None/empty values
        user_profile = clean_user_profile(user_profile)
        print(f"DEBUG - Cleaned user profile keys: {list(user_profile.keys())}")
        
        # If there are validation errors, return only the first one to avoid overwhelming the user
        if validation_errors:
            error_message = validation_errors[0]
            conversation.append({"role": "assistant", "content": error_message})
            
            # Calculate current missing fields
            current_missing = []
            for f in required_fields:
                if f not in user_profile or user_profile.get(f) is None:
                    if f == "Academic_Performance" and "Academic_Score" in user_profile and loan_type == "education":
                        continue
                    current_missing.append(f)
            
            print(f"DEBUG - Returning validation error: {error_message}")
            return MessageResponse(
                message=error_message,
                recorded={},
                missing_fields=current_missing
            )

        # Check completeness with enhanced logic
        missing_fields = []
        for f in required_fields:
            if f not in user_profile or user_profile.get(f) is None:
                # Special handling for derived fields
                if f == "Academic_Performance" and "Academic_Score" in user_profile and loan_type == "education":
                    print(f"DEBUG - {f} derived from Academic_Score, not missing")
                    continue
                missing_fields.append(f)
                print(f"DEBUG - Missing field: {f}")

        print(f"DEBUG - Final missing fields: {missing_fields}")
        print(f"DEBUG - User profile after processing: {list(user_profile.keys())}")

        # If complete -> run prediction and present result
        if not missing_fields:
            print("DEBUG - All fields collected, processing prediction...")

            try:
                # Convert string values to appropriate numeric types with enhanced error handling
                typed = user_profile.copy()
                
                # Get numeric fields based on loan type
                numeric_fields = []
                if loan_type == "education":
                    numeric_fields = ["Age", "Academic_Score", "Coapplicant_Income", "Guarantor_Networth", 
                                    "CIBIL_Score", "Loan_Term", "Expected_Loan_Amount"]
                elif loan_type == "home":
                    numeric_fields = ["Age", "Income", "Guarantor_income", "Tenure", 
                                    "CIBIL_score", "Down_payment", "Existing_total_EMI", 
                                    "Loan_amount_requested", "Property_value"]
                elif loan_type == "personal":
                    numeric_fields = ["Age", "Employment_Duration_Years", "Annual_Income", 
                                    "CIBIL_Score", "Existing_EMIs", "Loan_Term_Years", "Expected_Loan_Amount"]
                elif loan_type == "business":
                    numeric_fields = ["Business_Age_Years", "Annual_Revenue", "Net_Profit", "CIBIL_Score",
                                    "Existing_Loan_Amount", "Loan_Tenure_Years", "Expected_Loan_Amount"]
                elif loan_type == "gold":
                    numeric_fields = ["Age", "Annual_Income", "CIBIL_Score", "Gold_Value", "Loan_Amount", "Loan_Tenure"]
                elif loan_type == "car":
                    numeric_fields = ["Age", "applicant_annual_salary", "Coapplicant_Annual_Income", "CIBIL",
                                    "down_payment_percent", "Tenure", "loan_amount"]
                
                # Convert numeric fields with error handling
                conversion_errors = []
                for field in numeric_fields:
                    if field in typed:
                        try:
                            original_value = typed[field]
                            typed[field] = _to_float(typed[field])
                            print(f"DEBUG - Converted {field}: {original_value} -> {typed[field]}")
                        except Exception as e:
                            conversion_errors.append(f"Error converting {field}: {str(e)}")
                            print(f"DEBUG - Conversion error for {field}: {e}")

                if conversion_errors:
                    error_msg = "Data conversion errors: " + "; ".join(conversion_errors)
                    raise Exception(error_msg)

                # Create prediction input without customer fields
                prediction_input = {k: v for k, v in typed.items() 
                                  if not k.startswith("Customer_")}
                
                print(f"DEBUG - Prediction input for {loan_type}: {prediction_input}")
                
                # Make prediction with enhanced error handling
                try:
                    predicted_loan, predicted_interest = service.predict_loan(prediction_input)
                    print(f"DEBUG - Prediction successful: loan={predicted_loan}, interest={predicted_interest}")
                except Exception as pred_error:
                    print(f"DEBUG - Prediction failed: {pred_error}")
                    raise HTTPException(status_code=500, detail=f"Prediction error: {str(pred_error)}")

                # Get requested amount for summary based on loan type
                amount_field_mapping = {
                    "education": "Expected_Loan_Amount",
                    "home": "Loan_amount_requested", 
                    "personal": "Expected_Loan_Amount",
                    "business": "Expected_Loan_Amount",
                    "gold": "Loan_Amount",
                    "car": "loan_amount"
                }
                
                amount_field = amount_field_mapping.get(loan_type, "Expected_Loan_Amount")
                summary_requested_amount = int(typed.get(amount_field, 500000))
                print(f"DEBUG - Requested amount: {summary_requested_amount}")

                # Build summary with enhanced security
                if predicted_loan >= summary_requested_amount:
                    # Full approval - customer gets what they asked for
                    # SECURITY: Don't reveal maximum eligible amount
                    approved_amount = summary_requested_amount
                    approval_status = "APPROVED"
                    print(f"DEBUG - Full approval: {approved_amount}")
                else:
                    # Partial approval - show only what they can actually get
                    approved_amount = predicted_loan
                    approval_status = "PARTIAL_APPROVAL"
                    print(f"DEBUG - Partial approval: {approved_amount}")
                
                summary = {
                    "loan_type": loan_type,
                    "profile": {k: (int(v) if isinstance(v, float) and k in numeric_fields else v) 
                              for k, v in typed.items()},
                    "result": {
                        "approved_amount": int(approved_amount),
                        "interest_rate": float(predicted_interest),
                        "requested_amount": summary_requested_amount,
                        "status": approval_status
                    }
                }

                # Extract customer info from collected data
                customer_info = {
                    "name": typed.get("Customer_Name", "Unknown"),
                    "email": typed.get("Customer_Email", ""),
                    "phone": typed.get("Customer_Phone", "")
                }
                
                # Remove customer info from loan data for prediction
                loan_data_for_prediction = {k: v for k, v in typed.items() 
                                          if not k.startswith("Customer_")}
                
                # Save customer application data with error handling
                try:
                    file_path = storage_manager.save_customer_application(
                        loan_type=loan_type,
                        session_id=req.session_id,
                        customer_info=customer_info,
                        loan_data=loan_data_for_prediction,
                        prediction_result=summary
                    )
                    print(f"DEBUG - Customer application saved: {file_path}")
                except Exception as save_error:
                    print(f"WARNING: Failed to save customer data: {save_error}")

                # Reset user profile for new prediction but keep conversation
                SESSIONS[req.session_id]["user_profile"] = {}
                print("DEBUG - Reset user profile for new session")

                # Generate marketing-friendly response message
                customer_name = customer_info.get("name", "")
                loan_type_title = loan_type.title()
                
                if predicted_loan >= summary_requested_amount:
                    # Full approval message
                    assistant_msg = (
                        f"🎉 Fantastic news {customer_name}! You're PRE-APPROVED for your {loan_type_title} Loan!\n\n"
                        f"✅ YES! You are eligible for ₹{approved_amount:,} at {predicted_interest}% per annum\n\n"
                        f"🚀 What happens next:\n"
                        f"• Your loan is pre-approved and ready for processing\n"
                        f"• Competitive interest rate of {predicted_interest}% per annum\n"
                        f"• Fast-track processing with minimal documentation\n"
                        f"• Our relationship manager will contact you within 24 hours\n\n"
                        f"📞 We'll reach out to you at {customer_info.get('email', '')} or {customer_info.get('phone', '')} soon!"
                    )
                else:
                    # Partial approval message
                    assistant_msg = (
                        f"💡 Great news {customer_name}! You're ELIGIBLE for a {loan_type_title} Loan!\n\n"
                        f"✅ You can get ₹{approved_amount:,.0f} at {predicted_interest}% per annum\n\n"
                        f"🎯 Your loan offer:\n"
                        f"• Approved Amount: ₹{approved_amount:,.0f}\n"
                        f"• Interest Rate: {predicted_interest}% per annum\n"
                        f"• Pre-approved offer valid for 30 days\n"
                        f"• Flexible repayment options available\n\n"
                        f"💬 Want to discuss your loan requirements? Our specialist will call you!\n\n"
                        f"📞 We'll contact you at {customer_info.get('email', '')} or {customer_info.get('phone', '')} within 24 hours."
                    )
                
                conversation.append({"role": "assistant", "content": assistant_msg})
                print("DEBUG - Added success message to conversation")

                return MessageResponse(
                    message=assistant_msg,
                    recorded={},  # Don't show internal fields to user
                    missing_fields=[],
                    prediction=summary
                )
                
            except Exception as e:
                print(f"DEBUG - Prediction processing error: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        # If we have missing fields, ask for them with improved follow-up
        if missing_fields:
            print(f"DEBUG - Generating follow-up for missing fields: {missing_fields}")
            
            try:
                # Show progress by acknowledging what we have
                collected_fields = [f for f in required_fields if f in user_profile and user_profile.get(f) is not None]
                progress_msg = ""
                
                if recorded_now:
                    # Acknowledge newly recorded information
                    recorded_items = []
                    for k, v in recorded_now.items():
                        if k.startswith("Customer_"):
                            continue  # Don't show customer fields in progress
                        field_display = k.replace('_', ' ').title()
                        if isinstance(v, (int, float)) and v > 1000:
                            recorded_items.append(f"{field_display}: ₹{v:,.0f}")
                        else:
                            recorded_items.append(f"{field_display}: {v}")
                    
                    if recorded_items:
                        progress_msg = f"Great! I've recorded: {', '.join(recorded_items)}. "
                
                # Generate contextual follow-up
                followup = service.assistant_followup(conversation, user_profile, missing_fields)
                
                # Enhance follow-up with progress acknowledgment
                if progress_msg:
                    followup = progress_msg + followup
                
                conversation.append({"role": "assistant", "content": followup})
                print(f"DEBUG - Generated follow-up message: {followup[:100]}...")

                return MessageResponse(
                    message=followup,
                    recorded={},  # Don't show internal fields to user
                    missing_fields=missing_fields
                )
                
            except Exception as followup_error:
                print(f"DEBUG - Follow-up generation error: {followup_error}")
                # Fallback message
                fallback_msg = f"Thank you! I need a few more details to process your {loan_type} loan application. Could you please provide the missing information?"
                conversation.append({"role": "assistant", "content": fallback_msg})
                
                return MessageResponse(
                    message=fallback_msg,
                    recorded={},
                    missing_fields=missing_fields
                )
        else:
            # This should not happen as we handle complete cases above
            print("WARNING: No missing fields but prediction not processed")
            return MessageResponse(
                message="I have all the information. Let me process your loan application...",
                recorded={},
                missing_fields=[]
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"ERROR - Unexpected error in chat_message: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/session/{session_id}")
def get_session_info(session_id: str):
    """Get information about a chat session with enhanced details"""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = SESSIONS[session_id]
    service = LoanServiceFactory.get_service(state["loan_type"], OPENAI_API_KEY)
    
    # Calculate missing fields accounting for derived fields
    missing_fields = []
    collected_fields = list(state["user_profile"].keys())
    
    for f in service.get_required_fields():
        if f not in state["user_profile"]:
            # Handle derived fields
            if f == "Academic_Performance" and "Academic_Score" in state["user_profile"] and state["loan_type"] == "education":
                continue
            missing_fields.append(f)
    
    return {
        "session_id": session_id,
        "loan_type": state["loan_type"],
        "required_fields": service.get_required_fields(),
        "collected_fields": collected_fields,
        "missing_fields": missing_fields,
        "completion_percentage": round((len(collected_fields) / len(service.get_required_fields())) * 100, 2),
        "created_at": state["created_at"]
    }

@app.get("/admin/stats/{loan_type}")
def get_loan_stats(loan_type: str):
    """Get statistics for a specific loan type (admin endpoint)"""
    if loan_type not in LoanServiceFactory.get_available_loan_types():
        raise HTTPException(status_code=400, detail="Invalid loan type")
    
    try:
        stats = storage_manager.get_application_stats(loan_type)
        return {
            "loan_type": loan_type,
            "statistics": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/admin/stats")
def get_all_loan_stats():
    """Get statistics for all loan types (admin endpoint)"""
    try:
        all_stats = {}
        for loan_type in LoanServiceFactory.get_available_loan_types():
            all_stats[loan_type] = storage_manager.get_application_stats(loan_type)
        
        return all_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/admin/applications/{loan_type}")
def get_recent_applications(loan_type: str, limit: int = 10):
    """Get recent applications for a loan type (admin endpoint)"""
    if loan_type not in LoanServiceFactory.get_available_loan_types():
        raise HTTPException(status_code=400, detail="Invalid loan type")
    
    try:
        applications = storage_manager.get_customer_applications(loan_type, limit)
        return applications
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting applications: {str(e)}")

@app.get("/admin/exports")
def get_export_info():
    """Get information about available CSV exports (admin endpoint)"""
    try:
        from pathlib import Path
        import os
        
        export_info = {}
        for loan_type in LoanServiceFactory.get_available_loan_types():
            csv_path = Path("customer_data") / loan_type / "reports" / f"{loan_type}_applications.csv"
            
            if csv_path.exists():
                stat = csv_path.stat()
                # Count lines in CSV (excluding header)
                try:
                    with open(csv_path, 'r') as f:
                        record_count = sum(1 for line in f) - 1  # Subtract header
                except:
                    record_count = 0
                
                export_info[loan_type] = {
                    "exists": True,
                    "size": stat.st_size,
                    "lastModified": stat.st_mtime,
                    "recordCount": max(0, record_count)
                }
            else:
                export_info[loan_type] = {
                    "exists": False,
                    "size": 0,
                    "lastModified": None,
                    "recordCount": 0
                }
        
        return export_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting export info: {str(e)}")

@app.get("/admin/export/{loan_type}")
def download_csv_export(loan_type: str):
    """Download CSV export for a loan type (admin endpoint)"""
    if loan_type not in LoanServiceFactory.get_available_loan_types():
        raise HTTPException(status_code=400, detail="Invalid loan type")
    
    try:
        from fastapi.responses import FileResponse
        from pathlib import Path
        
        csv_path = Path("customer_data") / loan_type / "reports" / f"{loan_type}_applications.csv"
        
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")
        
        return FileResponse(
            path=str(csv_path),
            filename=f"{loan_type}_applications.csv",
            media_type='text/csv'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading CSV: {str(e)}")

@app.post("/admin/generate-report/{loan_type}")
def generate_csv_report(loan_type: str):
    """Generate a new CSV report for a loan type (admin endpoint)"""
    if loan_type not in LoanServiceFactory.get_available_loan_types():
        raise HTTPException(status_code=400, detail="Invalid loan type")
    
    try:
        # Generate CSV report using storage manager
        csv_path = storage_manager.export_to_csv(loan_type)
        
        return {
            "message": f"CSV report generated successfully for {loan_type} loans",
            "file_path": str(csv_path),
            "loan_type": loan_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("loan_app:app", host="0.0.0.0", port=8001, reload=True)