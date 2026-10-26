FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend, models, and data
COPY backend /app/backend
COPY models /app/models
COPY data /app/data

# Ensure runtime directories exist
RUN mkdir -p /app/data/uploads_tmp /app/data/apk_library

# Set environment variables
ENV PYTHONPATH=/app/backend \
    PORT=8000 \
    SECRET_KEY=WEThq0LDcxeXR4mzcx_wxTMDYgYtAEpM_gesjJ3aLtAIM8gozjiZzy49olHV2vHu \
    ALLOW_SYNTHETIC_MODEL=false \
    ENABLE_AI_EXPLANATIONS=true

WORKDIR /app/backend

# Initialize database and launch Uvicorn
EXPOSE 8000
CMD ["sh", "-c", "python -m app.cli init-db && python -m uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
