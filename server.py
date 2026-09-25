"""server.py - High-Performance Remote GPU Subtitle & Translation Server.

Run this script on any remote machine with an NVIDIA GPU to offload heavy Whisper
speech-to-text inference and multi-language translations from your laptop.

Usage on GPU Server:
    python server.py --model large-v3-turbo --device cuda --compute-type float16 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import platform
import sys
import time
from typing import Dict, List, Optional

# Automatically re-execute within local or user .venv if executed with system python
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    _curr_dir = os.path.dirname(os.path.abspath(__file__))
    _home_dir = os.path.expanduser("~")
    _candidates = [
        os.path.join(_curr_dir, ".venv", "bin", "python"),
        os.path.join(_curr_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(_home_dir, ".venv-lmls", "bin", "python"),
        os.path.join(_home_dir, ".venv", "bin", "python"),
    ]
    for _venv_py in _candidates:
        if os.path.isfile(_venv_py) and os.path.abspath(sys.executable) != os.path.abspath(_venv_py):
            _args = sys.orig_argv[1:] if hasattr(sys, "orig_argv") else sys.argv
            os.execv(_venv_py, [_venv_py] + _args)

try:
    import numpy as np
except ImportError:
    print(
        "\n❌ Missing required dependencies on this machine!\n"
        "   To install server requirements, run:\n"
        "     pip install --user fastapi uvicorn faster-whisper deep-translator torch numpy requests\n"
        "   Or with a virtual environment:\n"
        "     python3 -m venv ~/.venv-lmls\n"
        "     source ~/.venv-lmls/bin/activate\n"
        "     pip install fastapi uvicorn faster-whisper deep-translator torch numpy requests\n"
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [gpu_server]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_subtitles.server")

# Try importing FastAPI/Uvicorn, fallback to standard library http.server if needed
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


class AudioProcessRequest(BaseModel):
    audio_base64: str  # Base64 encoded float32 16kHz raw PCM bytes
    targets: List[str]
    whitelist: Optional[List[str]] = None
    sample_rate: int = 16000


class ServerState:
    model = None
    model_name: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    translation_engine = None


state = ServerState()


def load_gpu_model(model_name: str, device: str, compute_type: str) -> None:
    """Initialize faster-whisper on GPU with informative fallback."""
    from faster_whisper import WhisperModel
    from translator import TranslationEngine
    import ctranslate2

    cuda_count = 0
    try:
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception:
        pass

    if device == "cuda" and cuda_count == 0:
        logger.warning(
            "⚠️ Notice: No NVIDIA GPU detected on this machine ('%s').\n"
            "   Remember: server.py should be run ON YOUR REMOTE GPU SERVER where CUDA GPUs are present!\n"
            "   Falling back to CPU int8 mode for this host...",
            platform.node() if hasattr(platform, "node") else "local",
        )
        device = "cpu"
        compute_type = "int8"

    logger.info(
        f"Loading faster-whisper '{model_name}' on {device.upper()} with {compute_type}...\n"
        f"   Notice: If this is the first time loading '{model_name}', weights are downloading from Hugging Face (~1.5GB).\n"
        f"   Please wait for the download to finish (do NOT press Ctrl+C)!"
    )
    try:
        state.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        state.model_name = model_name
        state.device = device
        state.compute_type = compute_type
        state.translation_engine = TranslationEngine(engine_name="google")
        logger.info(f"Model successfully loaded on {device.upper()}!")
    except Exception as exc:
        logger.error(f"Failed to load model '{model_name}' on {device}: {exc}")
        err_str = str(exc)
        if "Connection reset by peer" in err_str or "ConnectError" in err_str or "104" in err_str:
            logger.error(
                "\n" + "=" * 70 + "\n"
                "❌ NETWORK / FIREWALL ERROR CONNECTING TO HUGGING FACE:\n"
                "   The server's outgoing internet connection to huggingface.co was reset.\n"
                "   Options to fix:\n"
                "   1. Use HF Mirror (if in China or restricted network):\n"
                "      export HF_ENDPOINT=https://hf-mirror.com\n"
                "      python3 server.py ...\n"
                "   2. Or download on your laptop and rsync/scp to the server:\n"
                "      rsync -avz ~/.cache/huggingface/ viegas@100.91.125.25:~/.cache/huggingface/\n"
                "   3. Or pass a local model path directly:\n"
                "      python3 server.py --model /path/to/model_dir\n"
                + "=" * 70 + "\n"
            )
            sys.exit(1)
        if device == "cuda":
            logger.warning("Falling back to CPU int8 mode...")
            state.model = WhisperModel(model_name, device="cpu", compute_type="int8")
            state.device = "cpu"
            state.compute_type = "int8"
            state.translation_engine = TranslationEngine(engine_name="google")
        else:
            raise exc


def process_audio_data(
    audio_array: np.ndarray,
    targets: List[str],
    whitelist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute GPU transcription and parallel translation for an audio chunk."""
    t0 = time.time()
    if state.model is None:
        raise RuntimeError("Model is not initialized.")

    import ctranslate2
    from transcriber import clean_transcribed_text
    from translator import normalize_lang_code

    whitelist_set = {w.lower().strip() for w in whitelist} if whitelist else set()
    norm_targets = [normalize_lang_code(t) for t in targets]

    # 1. Automatic Language Identification (Strictly restricted to Whitelist)
    detected_lang = None
    lang_prob = 1.0

    if whitelist_set:
        try:
            features = state.model.feature_extractor(audio_array)
            segment_features = np.zeros((features.shape[0], 3000), dtype=np.float32)
            usable_frames = min(features.shape[1], 3000)
            segment_features[:, :usable_frames] = features[:, :usable_frames]

            storage = ctranslate2.StorageView.from_array(
                np.expand_dims(segment_features, axis=0)
            )
            encoder_output = state.model.model.encode(storage)
            results = state.model.model.detect_language(encoder_output)

            candidates = []
            for token, prob in results[0]:
                code = token.strip("<|>").lower()
                if code in whitelist_set:
                    candidates.append((code, prob))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                detected_lang, lang_prob = candidates[0]
        except Exception as exc:
            logger.debug(f"GPU Language ID error: {exc}")
            detected_lang = next(iter(whitelist_set))

    # 2. Fast GPU Transcription
    segments, info = state.model.transcribe(
        audio_array,
        language=detected_lang,
        task="transcribe",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        without_timestamps=True,
    )

    full_text = " ".join([s.text for s in segments])
    cleaned = clean_transcribed_text(full_text)

    if not cleaned:
        return {
            "source_lang": detected_lang or "en",
            "original_text": "",
            "confidence": 0.0,
            "translations": {t: "" for t in norm_targets},
            "latency_ms": (time.time() - t0) * 1000.0,
        }

    spoken_lang = normalize_lang_code(detected_lang or info.language)

    # 3. Parallel Translation across targets
    import concurrent.futures

    translations: Dict[str, str] = {}
    futures_map = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(norm_targets)) as executor:
        for tgt in norm_targets:
            if tgt == spoken_lang or tgt.split("-")[0] == spoken_lang.split("-")[0]:
                translations[tgt] = cleaned
            else:
                fut = executor.submit(
                    state.translation_engine.translate_single, cleaned, spoken_lang, tgt
                )
                futures_map[fut] = tgt

        for fut in concurrent.futures.as_completed(futures_map):
            tgt = futures_map[fut]
            try:
                translations[tgt] = fut.result()
            except Exception:
                translations[tgt] = f"[{tgt.upper()}] {cleaned}"

    total_latency_ms = (time.time() - t0) * 1000.0

    logger.info(
        f"GPU processed in {total_latency_ms:.1f}ms [{spoken_lang.upper()}]: {cleaned}"
    )

    return {
        "source_lang": spoken_lang,
        "original_text": cleaned,
        "confidence": float(lang_prob if detected_lang else info.language_probability),
        "translations": translations,
        "latency_ms": total_latency_ms,
    }


if HAS_FASTAPI:
    app = FastAPI(title="Live Subtitles GPU Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "status": "online",
            "model": state.model_name,
            "device": state.device,
            "compute_type": state.compute_type,
        }

    @app.post("/process")
    def process_endpoint(req: AudioProcessRequest):
        try:
            # Decode base64 float32 PCM bytes
            raw_bytes = base64.b64decode(req.audio_base64)
            audio_array = np.frombuffer(raw_bytes, dtype=np.float32)
            return process_audio_data(audio_array, req.targets, req.whitelist)
        except Exception as exc:
            logger.error(f"Error processing request: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))


def parse_args():
    parser = argparse.ArgumentParser(description="Live Subtitles GPU Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Server port number.")
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3-turbo",
        help="Whisper model (e.g. small, medium, large-v3-turbo; default: large-v3-turbo).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device (cuda or cpu; default: cuda).",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16",
        help="Quantization (float16, bfloat16, int8; default: float16).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not HAS_FASTAPI:
        logger.error(
            "FastAPI and Uvicorn are required to run the server. Install via:\n"
            "pip install fastapi uvicorn"
        )
        sys.exit(1)

    load_gpu_model(args.model, args.device, args.compute_type)

    logger.info(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
