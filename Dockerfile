FROM vllm/vllm-openai:v0.4.3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENV HF_HOME=/app/model-cache

EXPOSE 8000

CMD bash -c "python3 -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 4096 \
    --tool-call-parser mistral \
    --enable-auto-tool-choice & \
    python3 -u handler.py"