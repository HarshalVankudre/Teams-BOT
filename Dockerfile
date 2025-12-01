FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY commands.py .
COPY rag/ ./rag/

# Environment variables are set in Cloud Run service configuration
# Do NOT copy .env file - use Cloud Run secrets/env vars instead

# Cloud Run uses PORT env variable (default 8080)
ENV PORT=8080
EXPOSE 8080

# Run the application
CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT
