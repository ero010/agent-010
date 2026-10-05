FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend.py index.html .env.example ./
CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT:-7860}"]
