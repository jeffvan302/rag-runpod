# rag-runpod

RunPod Serverless worker for BubbleAPI's optional chunking and embedding
accelerator.

The worker implements the protocol expected by BubbleAPI's setup CLI:

- `health`
- `preload_models`
- `embed_texts`
- `chunk_and_embed`

The image is intended to be published as:

```shell
ghcr.io/jeffvan302/rag-runpod-worker:0.2.0
```

## Local Checks

```shell
python -m unittest discover -s tests
docker build --platform linux/amd64 -t ghcr.io/jeffvan302/rag-runpod-worker:0.2.0 .
docker run --rm ghcr.io/jeffvan302/rag-runpod-worker:0.2.0 python src/handler.py --test_input '{"input":{"operation":"health","protocolVersion":1}}'
```

`preload_models` downloads the selected Hugging Face model into the attached
RunPod network volume and writes a manifest to
`/runpod-volume/bubble-rag/models/manifest.json`.
