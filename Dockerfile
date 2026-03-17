FROM vllm/vllm-openai:v0.4.3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN
ENV HF_HOME=/app/model-cache

# Download Mistral at build time — no cold start delay
RUN python -c "\
from huggingface_hub import login, snapshot_download; \
import os; \
login(token=os.environ['HF_TOKEN']); \
snapshot_download('mistralai/Mistral-7B-Instruct-v0.3')"

EXPOSE 8000

# vLLM starts first, then handler.py connects to it
CMD bash -c "python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 4096 \
    --tool-call-parser mistral \
    --enable-auto-tool-choice & \
    python -u handler.py"
```

---

**Folder structure:**
```
mistral-runpod/
├── Dockerfile
├── handler.py
└── requirements.txt