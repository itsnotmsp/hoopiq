FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc cmake && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "5_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
# cache bust Sun Apr 19 15:34:19 +07 2026
