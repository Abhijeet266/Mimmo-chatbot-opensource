# ✅ Most compatible — works on virtually all RunPod nodes
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-cudnn9-devel-ubuntu22.04

ENV VLLM_USE_CACHE=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update && apt-get install -y ninja-build curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

RUN uv venv $VIRTUAL_ENV

COPY requirements.txt /
RUN uv pip install vllm==0.10.1 && \
    uv pip install -r /requirements.txt

COPY rp_handler.py /

WORKDIR /
CMD ["python", "-u", "rp_handler.py"]