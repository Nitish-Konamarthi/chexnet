# CPU-only Dockerfile for Streamlit prototype
FROM python:3.9-slim

WORKDIR /app

# system deps for Pillow / opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire repo (app, DensenetModels.py, etc.)
COPY . /app

# Streamlit default port is 8501
EXPOSE 8501

ENV MODEL_PATH=/app/models/m-25012018-123527.pth.tar
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]