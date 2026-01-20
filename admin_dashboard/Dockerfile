FROM python:3.11-slim

WORKDIR /app

# Install dependencies (including openai for rag module imports)
RUN pip install --no-cache-dir flask psycopg2-binary python-dotenv gunicorn openai tiktoken pinecone

# Copy admin dashboard code
COPY admin_dashboard/ /app/

# Copy rag module for admin_logger
COPY rag/ /app/rag/

# Expose port
EXPOSE 8080

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "dashboard:app"]
