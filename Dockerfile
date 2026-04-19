FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for XGBoost
RUN apt-get update && apt-get install -y gcc cmake && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port
EXPOSE $PORT

# Start the API
CMD uvicorn 5_api_server:app --host 0.0.0.0 --port ${PORT:-8000}
