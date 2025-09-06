from typing import Dict, Optional, Type
import os
from .education_loan import EducationLoanService
from .home_loan import HomeLoanService
from .personal_loan import PersonalLoanService
from .gold_loan import GoldLoanService
from .business_loan import BusinessLoanService
from .car_loan import CarLoanService
from .base_loan import BaseLoanService

class LoanServiceFactory:
    """Factory class to create appropriate loan service instances"""
    
    _services: Dict[str, BaseLoanService] = {}
    
    # Define service mappings
    SERVICE_CLASSES: Dict[str, Type[BaseLoanService]] = {
        "education": EducationLoanService,
        "home": HomeLoanService,
        "personal": PersonalLoanService,
        "gold": GoldLoanService,
        "business": BusinessLoanService,
        "car": CarLoanService
    }
    
    MODEL_PATHS: Dict[str, str] = {
        "education": "models/education_loan_models",
        "home": "models/home_loan_models", 
        "personal": "models/personal_loan_models",
        "gold": "models/gold_loan_models",
        "business": "models/business_loan_models",
        "car": "models/car_loan_models"
    }
    
    @classmethod
    def get_service(cls, loan_type: str, openai_api_key: Optional[str] = None) -> BaseLoanService:
        """Get loan service instance for the specified loan type"""
        
        # Normalize loan type
        loan_type = loan_type.lower().strip()
        
        if loan_type not in cls.SERVICE_CLASSES:
            raise ValueError(f"Unsupported loan type: {loan_type}. Available types: {list(cls.SERVICE_CLASSES.keys())}")
        
        # Use cached instance if available
        if loan_type not in cls._services:
            cls._services[loan_type] = cls._create_service(loan_type, openai_api_key)
        
        return cls._services[loan_type]
    
    @classmethod
    def _create_service(cls, loan_type: str, openai_api_key: Optional[str] = None) -> BaseLoanService:
        """Create a new loan service instance"""
        
        service_class = cls.SERVICE_CLASSES[loan_type]
        model_path = cls.MODEL_PATHS[loan_type]
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            print(f"WARNING: Model path '{model_path}' does not exist. Service will be created but models may not load.")
        
        try:
            return service_class(model_path, openai_api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to create {loan_type} loan service: {str(e)}")
    
    @classmethod
    def get_available_loan_types(cls) -> list:
        """Get list of available loan types"""
        return list(cls.SERVICE_CLASSES.keys())
    
    @classmethod
    def get_service_info(cls, loan_type: str) -> Dict[str, str]:
        """Get information about a specific loan service"""
        loan_type = loan_type.lower().strip()
        
        if loan_type not in cls.SERVICE_CLASSES:
            raise ValueError(f"Unsupported loan type: {loan_type}")
        
        service_class = cls.SERVICE_CLASSES[loan_type]
        model_path = cls.MODEL_PATHS[loan_type]
        
        return {
            "loan_type": loan_type,
            "service_class": service_class.__name__,
            "model_path": model_path,
            "model_exists": os.path.exists(model_path)
        }
    
    @classmethod
    def clear_cache(cls, loan_type: Optional[str] = None):
        """Clear cached service instances - specific type or all"""
        if loan_type:
            if loan_type in cls._services:
                del cls._services[loan_type]
        else:
            cls._services.clear()
    
    @classmethod
    def reload_service(cls, loan_type: str, openai_api_key: Optional[str] = None) -> BaseLoanService:
        """Force reload of a specific service"""
        cls.clear_cache(loan_type)
        return cls.get_service(loan_type, openai_api_key)