FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/runpod-volume/huggingface
ENV HF_HUB_CACHE=/runpod-volume/huggingface/hub
ENV BUBBLE_RAG_MODEL_MANIFEST=/runpod-volume/bubble-rag/models/manifest.json

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
  && python -m pip install -r requirements.txt

COPY src ./src

CMD ["python", "-u", "src/handler.py"]
