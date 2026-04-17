# 🔹 Use official PyTorch image with CUDA (better fit than Ultralytics)
FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 🔹 Install system deps
RUN apt-get update && apt-get install -y \
    python3-venv \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 🔹 Create virtual environment
RUN python3 -m venv /opt/venv

# 🔹 Activate venv
ENV PATH="/opt/venv/bin:$PATH"

# 🔹 Upgrade pip inside venv
RUN pip install --upgrade pip

# 🔹 Install Python deps
RUN pip install --no-cache-dir \
    celery \
    redis \
    opencv-python-headless \
    pydantic \
    pydantic_settings \
    pillow \
    torchvision

# 🔹 Copy app
COPY ./app ./app

CMD ["celery", "-A", "app.services.celery_client", "worker", "-Q", "classification", "--loglevel=info", "-E"]