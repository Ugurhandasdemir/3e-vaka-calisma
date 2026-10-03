FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake backbone weights into the image so restarts do not re-download them.
RUN python -c "import torchvision; torchvision.models.wide_resnet50_2(weights='IMAGENET1K_V1')"

COPY . .

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    DATA_DIR=/data

EXPOSE 7860

CMD ["python", "app.py"]
