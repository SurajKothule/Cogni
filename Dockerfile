# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application files
COPY loan_app.py .
COPY loan_services/ ./loan_services/
COPY customer_data/ ./customer_data/
COPY .env .

# Create necessary directories
RUN mkdir -p customer_data/education/applications customer_data/education/reports \
    customer_data/home/applications customer_data/home/reports \
    customer_data/personal/applications customer_data/personal/reports \
    customer_data/gold/applications customer_data/gold/reports \
    customer_data/business/applications customer_data/business/reports \
    customer_data/car/applications customer_data/car/reports

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:$PORT/health')" || exit 1

# Run the application
CMD uvicorn loan_app:app --host 0.0.0.0 --port $PORT