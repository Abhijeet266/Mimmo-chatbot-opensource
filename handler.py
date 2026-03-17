import os
import json
import logging
import time
import requests
import sys

import runpod
from huggingface_hub import login

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
VLLM_HOST  = "http://localhost:8000"


def wait_for_vllm(timeout=300) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"{VLLM_HOST}/health", timeout=5).status_code == 200:
                logger.info("vLLM is ready")
                return True
        except Exception:
            pass
        logger.info(f"Waiting for vLLM... ({int(time.time()-start)}s)")
        time.sleep(5)
    return False


def handler(event):
    try:
        input_data = event.get("input", {})
        messages   = input_data.get("messages")

        if not messages:
            return {"error": "Missing 'messages' in input"}

        payload = {
            "model":       input_data.get("model", MODEL_NAME),
            "messages":    messages,
            "temperature": input_data.get("temperature", 0.7),
            "max_tokens":  input_data.get("max_tokens", 512),
        }

        # Include tools if sent — needed for get_risk_evaluation
        if "tools" in input_data:
            payload["tools"]       = input_data["tools"]
            payload["tool_choice"] = input_data.get("tool_choice", "auto")

        response = requests.post(
            f"{VLLM_HOST}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )

        if not response.ok:
            return {"error": f"vLLM Error ({response.status_code}): {response.text}"}

        result  = response.json()
        message = result["choices"][0]["message"]

        return {
            "choices": [{
                "message": {
                    "role":       "assistant",
                    "content":    message.get("content", None),
                    "tool_calls": message.get("tool_calls", None),
                }
            }],
            "usage": result.get("usage", {})
        }

    except Exception as e:
        logger.exception(f"Handler error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        logger.info("HuggingFace login successful")

    if not wait_for_vllm():
        logger.error("vLLM failed to start")
        sys.exit(1)

    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})