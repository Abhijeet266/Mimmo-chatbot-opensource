#!/bin/bash
set -e

echo "Logging into HuggingFace..."
python3 -c "from huggingface_hub import login; import os; login(token=os.environ.get('HF_TOKEN', ''))" || echo "HF login skipped"

echo "Starting vLLM server..."
python3 -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 4096 \
    --tool-call-parser mistral \
    --enable-auto-tool-choice &

echo "Starting RunPod handler..."
python3 -u handler.py