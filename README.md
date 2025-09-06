# 🏦 CogniBank Loan Application API

A production-ready FastAPI backend for AI-powered loan applications with optional MongoDB integration and local fallback storage.

## 🚀 Features

- **6 Loan Types**: Education, Home, Personal, Gold, Business, Car
- **AI-Powered Chat**: OpenAI integration for natural conversations
- **MongoDB Storage (optional)**: Uses MongoDB Atlas when available, falls back to local storage
- **Admin Dashboard (CLI)**: View stats, recent applications, and CSV export locations
- **ML Predictions**: Instant loan eligibility and interest rate calculations
- **RESTful API**: Clean endpoints for frontend integration

## ⚡ Quickstart (Backend)

1. Create and activate a virtual environment
   - Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file (see variables below)
4. Run the API (default local port: 8001)
   ```bash
   uvicorn loan_app:app --host 0.0.0.0 --port 8001 --reload
   ```
5. Open API docs: http://localhost:8001/docs

## 🔧 Environment Variables

```bash
# Required for AI chat
OPENAI_API_KEY=your_openai_api_key

# Optional: enable MongoDB storage (falls back to local if not set or connection fails)
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=loan_applications

# Port is controlled by your process manager (uvicorn/fly). For local run use 8001.
```

- If MongoDB connection fails, the app automatically uses local storage under `customer_data/`.

## 📋 API Endpoints (Core)

- `GET /health` — Health check
- `GET /loan-types` — List available loan types and descriptions
- `POST /chat/start` — Start a loan application chat session
- `POST /chat/message` — Continue chat in an existing session

Note: Admin features are provided via a local CLI (`admin_dashboard.py`), not HTTP endpoints.

## 🛠️ Admin Dashboard (CLI)

View statistics, recent applications, and CSV export paths using the interactive CLI:

```bash
python admin_dashboard.py
```

- CSV exports are written under `customer_data/<loan_type>/reports/`.

## 🗂️ Project Structure

```
├── loan_services/               # Loan service modules
│   ├── base_loan.py            # Abstract base class
│   ├── education_loan.py       # Education loan logic
│   ├── home_loan.py            # Home loan logic
│   ├── personal_loan.py        # Personal loan logic
│   ├── gold_loan.py            # Gold loan logic
│   ├── business_loan.py        # Business loan logic
│   ├── car_loan.py             # Car loan logic
│   └── loan_factory.py         # Service factory
├── customer_data/               # Data management (local + MongoDB integration)
│   ├── storage_manager.py      # Local storage
│   └── mongodb_storage_manager.py # MongoDB storage (optional)
├── models/                      # ML models by loan type
├── loan_app.py                  # Main FastAPI application
├── admin_dashboard.py           # Admin CLI for insights/exports
├── requirements.txt             # Python dependencies
├── Banking-Marketing-master/    # Next.js frontend (optional)
├── Dockerfile                   # Container configuration
├── fly.toml                     # Fly.io deployment config
├── SETUP.md                     # Detailed setup (backend + frontend)
└── DEPLOYMENT.md                # Deployment instructions
```

## 🎨 Frontend (Optional)

A Next.js frontend is available in `Banking-Marketing-master/`. See `SETUP.md` for instructions to run it against this API (defaults to `http://localhost:8001`).

## 🚀 Deployment

This application is ready for Fly.io. See `DEPLOYMENT.md` for complete instructions.

```bash
flyctl deploy --app cognibank-api
```

- Local default port is 8001. In containers/platforms, use the platform-provided port (e.g., Fly.io may map to 8080 internally).

## 🔒 Security

- Keep secrets in `.env` files and never commit them
- Validate inputs (handled via Pydantic models and service validation)
- Use HTTPS in production and secure your MongoDB cluster

## 📞 Support

- Check `SETUP.md` and `DEPLOYMENT.md`
- Review API docs at `/docs`
- Open an issue with logs and steps to reproduce