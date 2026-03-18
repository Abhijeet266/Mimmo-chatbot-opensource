FROM vllm/vllm-openai:v0.6.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENV HF_HOME=/app/model-cache

EXPOSE 8000

# Override the base image's entrypoint — this is the fix
ENTRYPOINT []
CMD ["/bin/bash", "start.sh"]

