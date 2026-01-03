# Extends official Ultralytics Docker image for YOLO11
FROM ultralytics/ultralytics:latest-cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/best.onnx

# Install FastAPI and dependencies
RUN pip install --no-cache-dir fastapi[all] uvicorn[standard] loguru onnxruntime

WORKDIR /app

# Create directory structure
RUN mkdir -p /app/models /app/logs

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Copy your ONNX model
COPY models/ /app/models/

# Install the application package
RUN pip install --no-cache-dir -e .

# Set PYTHONPATH to include the src directory
ENV PYTHONPATH=/app/src

# Port for Vertex AI
EXPOSE 8080

# Start the inference server
ENTRYPOINT ["python", "src/main.py"]