# Use the official Ultralytics image (highly optimized for YOLO)
FROM ultralytics/ultralytics:latest-python

WORKDIR /app

# Install Torch with CUDA support
RUN pip install --no-cache-dir --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install only the web server parts (YOLO is already installed)
RUN pip install --no-cache-dir \
    celery \
    redis \
    opencv-python-headless \
    pydantic \
    pydantic_settings

COPY ./app ./app

CMD ["celery", "-A", "app.services.celery_client", "worker", "-Q", "detection", "--loglevel=info"]