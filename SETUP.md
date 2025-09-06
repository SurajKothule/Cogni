# 🚀 Loan Management System - Setup Guide

## 📋 Prerequisites

- Python 3.8+ installed
- Node.js 16+ installed
- OpenAI API key (get from [OpenAI Platform](https://platform.openai.com/api-keys))

## 🔧 Backend Setup (FastAPI)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd loan-management-system
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file and add your OpenAI API key
# .env file should contain:
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 5. Start the Backend Server

```bash
# Option 1: Using the start script
python start_app.py

# Option 2: Using uvicorn directly
uvicorn loan_app:app --host 0.0.0.0 --port 8001 --reload

# Option 3: Using the integrated app
python start_integrated_app.py
```

The backend will be available at: `http://localhost:8001`

## 🎨 Frontend Setup (Next.js)

### 1. Navigate to Frontend Directory

```bash
cd Banking-Marketing-master
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Frontend Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env.local

# Edit .env.local file:
NEXT_PUBLIC_API_URL=http://localhost:8001
OPENAI_API_KEY=your_actual_openai_api_key_here
NEXT_PUBLIC_APP_NAME=NextBank
NEXT_PUBLIC_APP_VERSION=1.0.0
```

### 4. Start the Frontend Development Server

```bash
npm run dev
```

The frontend will be available at: `http://localhost:3000`

## 🗂️ Project Structure

```
loan-management-system/
├── loan_services/          # Loan processing services
│   ├── base_loan.py        # Base loan service class
│   ├── business_loan.py    # Business loan service
│   ├── education_loan.py   # Education loan service
│   ├── home_loan.py        # Home loan service
│   ├── personal_loan.py    # Personal loan service
│   └── gold_loan.py        # Gold loan service
├── models/                 # ML models for loan prediction
├── customer_data/          # Customer data management
├── Banking-Marketing-master/ # Next.js frontend
├── loan_app.py            # Main FastAPI application
├── admin_dashboard.py     # Admin dashboard
└── requirements.txt       # Python dependencies
```

## 🤖 Available Loan Types

1. **Education Loan** - For students and educational expenses
2. **Home Loan** - For property purchase and construction
3. **Personal Loan** - For personal financial needs
4. **Business Loan** - For business expansion and operations
5. **Gold Loan** - Against gold collateral

## 🔒 Security Notes

### ⚠️ IMPORTANT: Never commit sensitive files to Git!

The following files contain sensitive information and are excluded from Git:

- `.env` - Contains API keys and secrets
- `Banking-Marketing-master/.env` - Frontend environment variables
- `customer_data/*/applications/` - Customer application data
- `models/*.pkl` - ML model files (optional)

### 🛡️ Environment Variables

Always use environment variables for:

- API keys (OpenAI, Stripe, etc.)
- Database URLs
- Secret keys
- Third-party service credentials

## 🧪 Testing

### Backend Testing

```bash
# Test the API endpoints
python test_backend_api.py

# Test individual loan services
python -m pytest tests/ -v
```

### Frontend Testing

```bash
cd Banking-Marketing-master
npm test
```

## 📊 Admin Dashboard

Access the admin dashboard at: `http://localhost:8001/admin`

Features:

- View loan applications
- Generate reports
- Monitor system performance
- Manage customer data

## 🚀 Production Deployment

### Backend Deployment

1. Set up a production server (AWS, GCP, Azure, etc.)
2. Install Python and dependencies
3. Set environment variables securely
4. Use a production WSGI server like Gunicorn
5. Set up reverse proxy with Nginx

### Frontend Deployment

1. Build the Next.js application: `npm run build`
2. Deploy to Vercel, Netlify, or your preferred hosting
3. Set production environment variables
4. Configure API endpoints for production

## 🔧 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**

   - Ensure your API key is valid and has sufficient credits
   - Check that the key is properly set in the .env file

2. **Port Already in Use**

   - Change the port in the startup scripts
   - Kill existing processes using the port

3. **Module Not Found Errors**

   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **Frontend API Connection Issues**
   - Verify backend is running on the correct port
   - Check CORS settings in the FastAPI application
   - Ensure API URL is correct in frontend .env file

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the code documentation
3. Create an issue in the repository

## 🎯 Next Steps

After setup:

1. Test all loan types through the frontend
2. Review the admin dashboard
3. Customize the ML models for your use case
4. Add additional loan types if needed
5. Implement additional security measures for production
