FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc cmake && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENTRYPOINT ["sh", "-c", "uvicorn 5_api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
