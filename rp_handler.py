import os
import logging
import time
from typing import Dict, List
from dataclasses import dataclass

import torch
import runpod
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ─────────────────────────────────────────────
#  CONFIG
#
# ─────────────────────────────────────────────
@dataclass
class Config:
    MODEL_NAME:             str   = "mistralai/Mistral-7B-Instruct-v0.3"
    DTYPE:                  str   = "bfloat16"
    GPU_MEMORY_UTILIZATION: float = 0.90
    MAX_MODEL_LEN:          int   = 4096
    GPU_TOTAL_MEMORY:       float = 48.0    # A40
    MAX_INPUT_TOKENS:       int   = 3800    # leaves room for generation
    DEFAULT_MAX_TOKENS:     int   = 512
    DEFAULT_TEMPERATURE:    float = 0.7
    DEFAULT_TOP_P:          float = 0.9

config = Config()


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  GPU MONITORING
# ─────────────────────────────────────────────
def log_gpu_memory():
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved  = torch.cuda.memory_reserved()  / 1024**3
            logger.info(f"GPU — Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
        except Exception as e:
            logger.warning(f"Could not get GPU memory: {e}")


def log_detailed_gpu_state():
    if torch.cuda.is_available():
        try:
            allocated     = torch.cuda.memory_allocated()    / 1024**3
            reserved      = torch.cuda.memory_reserved()     / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            free          = config.GPU_TOTAL_MEMORY - reserved
            utilization   = (reserved / config.GPU_TOTAL_MEMORY) * 100

            logger.info("=" * 60)
            logger.info("GPU Memory State:")
            logger.info(f"  Total     : {config.GPU_TOTAL_MEMORY:.2f}GB")
            logger.info(f"  Allocated : {allocated:.2f}GB")
            logger.info(f"  Reserved  : {reserved:.2f}GB ({utilization:.1f}%)")
            logger.info(f"  Peak      : {max_allocated:.2f}GB")
            logger.info(f"  Free      : {free:.2f}GB")
            if utilization > 85:
                logger.warning(f"HIGH GPU utilization: {utilization:.1f}%")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Could not get detailed GPU state: {e}")


# ─────────────────────────────────────────────
#  GLOBAL RESOURCES — singleton, loads once
# ─────────────────────────────────────────────
class GlobalResources:
    _instance    = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load()
            GlobalResources._initialized = True

    def _load(self):
        logger.info("Initializing global resources...")

        try:
            log_detailed_gpu_state()

            # HuggingFace login — required for Mistral (gated model)
            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                from huggingface_hub import login
                login(token=hf_token)
                logger.info("HuggingFace login successful.")
            else:
                logger.warning("HF_TOKEN not set — Mistral 7B may fail to download.")

            # Tokenizer
            logger.info(f"Loading tokenizer: {config.MODEL_NAME}")
            t0 = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            logger.info(f"Tokenizer loaded in {time.time()-t0:.2f}s")

            # vLLM model
            logger.info(f"Loading vLLM model: {config.MODEL_NAME}")
            t0 = time.time()
            self.llm = LLM(
                model=config.MODEL_NAME,
                dtype=config.DTYPE,
                trust_remote_code=True,
                gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
                max_model_len=config.MAX_MODEL_LEN,
                enable_prefix_caching=True,
            )
            logger.info(f"vLLM model loaded in {time.time()-t0:.2f}s")

            log_detailed_gpu_state()
            self.initialized = True
            logger.info("All resources ready.")

        except Exception as e:
            logger.error(f"Resource initialization failed: {e}", exc_info=True)
            log_detailed_gpu_state()
            self.tokenizer   = None
            self.llm         = None
            self.initialized = False
            # DO NOT raise — keep RunPod worker alive


# Boot on import
resources = None
try:
    resources = GlobalResources()
    if not resources.initialized:
        logger.critical("Resources failed to init — worker staying alive.")
        resources = None
except Exception as e:
    logger.critical(f"GlobalResources raised: {e}", exc_info=True)
    resources = None


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def count_tokens(text: str) -> int:
    try:
        if resources is None or resources.tokenizer is None:
            return int(len(text.split()) * 1.3)
        return len(resources.tokenizer.encode(text))
    except Exception as e:
        logger.error(f"Token count error: {e}")
        return int(len(text.split()) * 1.3)


def validate_input(input_data: Dict) -> None:
    if not isinstance(input_data, dict):
        raise ValueError("Input must be a JSON object.")

    has_prompt   = isinstance(input_data.get("prompt"),   str)  and input_data["prompt"].strip()
    has_messages = isinstance(input_data.get("messages"), list) and len(input_data["messages"]) > 0

    if not has_prompt and not has_messages:
        raise ValueError("Provide 'prompt' (string) or 'messages' (list) in input.")

    if has_messages:
        for m in input_data["messages"]:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError("Each message needs 'role' and 'content' keys.")
            if m["role"] not in ("system", "user", "assistant"):
                raise ValueError(f"Invalid role '{m['role']}'. Use: system, user, assistant.")


# ─────────────────────────────────────────────
#  CORE INFERENCE
# ─────────────────────────────────────────────
def run_inference(
    messages:    List[Dict],
    max_tokens:  int,
    temperature: float,
    top_p:       float,
) -> Dict:

    # Apply Mistral chat template
    formatted_prompt = resources.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Token safety check
    prompt_tokens = count_tokens(formatted_prompt)
    logger.info(f"Prompt tokens after template: {prompt_tokens}")

    if prompt_tokens > config.MAX_INPUT_TOKENS:
        raise ValueError(
            f"Input too large: {prompt_tokens} tokens "
            f"(max {config.MAX_INPUT_TOKENS})."
        )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    logger.info("Running vLLM inference...")
    log_gpu_memory()

    t0      = time.time()
    outputs = resources.llm.generate([formatted_prompt], sampling_params)
    elapsed = time.time() - t0

    logger.info(f"Inference done in {elapsed:.2f}s")

    generated_text  = outputs[0].outputs[0].text.strip()
    completion_toks = len(outputs[0].outputs[0].token_ids)

    return {
        "generated_text":  generated_text,
        "finish_reason":   outputs[0].outputs[0].finish_reason,
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_toks,
            "total_tokens":      prompt_tokens + completion_toks,
        },
        "inference_time": round(elapsed, 2),
    }


# ─────────────────────────────────────────────
#  RUNPOD HANDLER
# ─────────────────────────────────────────────
def handler(event: Dict) -> Dict:
    """
    Accepts:

    Simple prompt:
    {
        "input": {
            "prompt": "What is Mistral 7B?",
            "max_tokens": 512,       <- optional
            "temperature": 0.7,      <- optional
            "top_p": 0.9             <- optional
        }
    }

    Full chat history:
    {
        "input": {
            "messages": [
                {"role": "system",    "content": "You are helpful."},
                {"role": "user",      "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
                {"role": "user",      "content": "Explain vLLM"}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    Returns:
    {
        "status":          "success",
        "generated_text":  "...",
        "finish_reason":   "stop",
        "usage":           {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N},
        "inference_time":  1.23,
        "processing_time": 1.45
    }
    """
    request_id = event.get("id", "unknown")
    start_time = time.time()

    try:
        logger.info("=" * 60)
        logger.info(f"NEW REQUEST — ID: {request_id}")
        logger.info("=" * 60)
        log_gpu_memory()

        # Resource guard
        if resources is None or not resources.initialized:
            return {
                "status":     "error",
                "error":      "Model not initialized. Check GPU / HF_TOKEN.",
                "error_type": "InitializationError",
            }

        if resources.llm is None or resources.tokenizer is None:
            return {
                "status":     "error",
                "error":      "Model components unavailable.",
                "error_type": "InitializationError",
            }

        # Validate
        input_data = event.get("input", {})
        validate_input(input_data)

        # Build messages
        if "messages" in input_data:
            messages = input_data["messages"]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": input_data["prompt"]},
            ]

        max_tokens  = int(  input_data.get("max_tokens",  config.DEFAULT_MAX_TOKENS))
        temperature = float(input_data.get("temperature", config.DEFAULT_TEMPERATURE))
        top_p       = float(input_data.get("top_p",       config.DEFAULT_TOP_P))

        logger.info(f"max_tokens={max_tokens} temperature={temperature} top_p={top_p}")

        result = run_inference(messages, max_tokens, temperature, top_p)

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log_gpu_memory()

        processing_time = time.time() - start_time
        logger.info(f"Request {request_id} done in {processing_time:.2f}s")
        logger.info("=" * 60)

        return {
            "status":          "success",
            "generated_text":  result["generated_text"],
            "finish_reason":   result["finish_reason"],
            "usage":           result["usage"],
            "inference_time":  result["inference_time"],
            "processing_time": round(processing_time, 2),
        }

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM — {request_id}: {e}")
        log_detailed_gpu_state()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status":          "error",
            "error":           "GPU out of memory. Reduce max_tokens or input size.",
            "error_type":      "OutOfMemoryError",
            "processing_time": round(time.time() - start_time, 2),
        }

    except RuntimeError as e:
        err = str(e)
        logger.error(f"RuntimeError — {request_id}: {err}")
        log_detailed_gpu_state()
        if "cuda" in err.lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status":          "error",
            "error":           f"Runtime error: {err[:300]}",
            "error_type":      "RuntimeError",
            "processing_time": round(time.time() - start_time, 2),
        }

    except ValueError as e:
        logger.error(f"ValidationError — {request_id}: {e}")
        return {
            "status":          "error",
            "error":           str(e),
            "error_type":      "ValidationError",
            "processing_time": round(time.time() - start_time, 2),
        }

    except Exception as e:
        logger.exception(f"UnexpectedError — {request_id}: {e}")
        log_detailed_gpu_state()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return {
            "status":          "error",
            "error":           f"Unexpected error: {str(e)[:300]}",
            "error_type":      "UnexpectedError",
            "processing_time": round(time.time() - start_time, 2),
        }


# ─────────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────────
def health_check(event: Dict) -> Dict:
    try:
        if resources is None or not resources.initialized:
            return {"status": "unhealthy", "message": "Resources not initialized."}
        if resources.llm is None or resources.tokenizer is None:
            return {"status": "unhealthy", "message": "Model components missing."}
        log_gpu_memory()
        return {
            "status":          "healthy",
            "model":           config.MODEL_NAME,
            "dtype":           config.DTYPE,
            "gpu_memory_util": config.GPU_MEMORY_UTILIZATION,
            "max_model_len":   config.MAX_MODEL_LEN,
        }
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Mistral 7B Instruct — RunPod Serverless")
    logger.info(f"Model   : {config.MODEL_NAME}")
    logger.info(f"dtype   : {config.DTYPE}")
    logger.info(f"GPU     : {config.GPU_TOTAL_MEMORY}GB A40")
    logger.info(f"Ready   : {resources is not None and resources.initialized}")
    logger.info("=" * 60)

    if resources and resources.initialized:
        log_detailed_gpu_state()

    runpod.serverless.start({
        "handler":      handler,
        "health_check": health_check,
    })