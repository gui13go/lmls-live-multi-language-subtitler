"""transcriber.py - Whisper model management, automatic whitelisted language identification,

and low-latency speech transcription using faster-whisper (CTranslate2).
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple

import numpy as np

from audio import AudioSegment

logger = logging.getLogger("live_subtitles.transcriber")


@dataclass
class TranscriptionResult:
    """Output from the speech-to-text pipeline."""

    text: str
    language: str
    confidence: float
    timestamp: float
    duration_s: float


# Common Whisper hallucination strings to filter out
HALLUCINATION_PATTERNS = [
    r"^\s*$",
    r"^[\.\,\?\!\s]+$",
    r"^[Mm]+[\.\,\!\s]*$",
    r"^[Uu]+[hm]+[\.\,\!\s]*$",
    r"^(thank you|thanks for watching|subscribe|subtitles by|copyright|transcription by)[\.\!\s]*$",
    r"^(you|bye|hello)[\.\!\s]*$",
]
HALLUCINATION_REGEXES = [
    re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS
]


def clean_transcribed_text(text: str) -> Optional[str]:
    """Sanitize transcribed text and filter empty or hallucinated outputs.

    Returns:
        Cleaned text string, or None if the text should be discarded.
    """
    cleaned = text.strip()
    if not cleaned:
        return None

    # Check against known hallucination phrases
    for regex in HALLUCINATION_REGEXES:
        if regex.match(cleaned):
            return None

    # Filter excessive repetition of single characters (e.g., '...........' or 'aaaaa')
    if len(cleaned) > 5 and len(set(cleaned.replace(" ", ""))) <= 2:
        return None

    return cleaned


class WhisperTranscriber(threading.Thread):
    """Manages the faster-whisper CTranslate2 model, consumes speech segments from

    an input queue, performs whitelisted language detection, and outputs transcription results.
    Can run locally or offload inference to a remote GPU server.
    """

    def __init__(
        self,
        audio_queue: queue.Queue[AudioSegment],
        result_callback: Callable[[TranscriptionResult], None],
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
        whitelist_languages: Optional[List[str]] = None,
        target_languages: Optional[List[str]] = None,
        server_url: Optional[str] = None,
        bundle_callback: Optional[Callable[[Any], None]] = None,
        mock_mode: bool = False,
    ):
        super().__init__(daemon=True, name="TranscriberThread")
        self.audio_queue = audio_queue
        self.result_callback = result_callback
        self.bundle_callback = bundle_callback
        self.target_languages = [t.strip().lower() for t in target_languages] if target_languages else []
        self.server_url = server_url.rstrip("/") if server_url else None
        self.model_size = model_size
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.mock_mode = mock_mode

        # Whitelist processing: clean and lowercase language codes (e.g., ['en', 'zh', 'de'])
        if whitelist_languages:
            self.whitelist: Set[str] = {
                lang.strip().lower() for lang in whitelist_languages if lang.strip()
            }
        else:
            self.whitelist = set()

        self._stop_event = threading.Event()
        self.model = None
        self.resolved_device = "cpu"
        self.resolved_compute_type = "int8"
        self._last_server_error_log = 0.0

        if self.server_url:
            self._verify_remote_server()
        elif not self.mock_mode:
            self._load_model()

    def _verify_remote_server(self) -> None:
        """Verify remote GPU server health."""
        import requests

        logger.info(f"Connecting to remote GPU server at {self.server_url}...")
        try:
            r = requests.get(f"{self.server_url}/health", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                logger.info(
                    f"Connected to remote GPU server! Model: {data.get('model')}, "
                    f"Device: {data.get('device')}, Compute: {data.get('compute_type')}"
                )
            else:
                logger.warning(f"Server responded with status {r.status_code}.")
        except Exception as exc:
            err_msg = str(exc)
            if "Connection refused" in err_msg or "111" in err_msg:
                logger.error(
                    f"⚠️ Remote GPU server at {self.server_url} refused connection!\n"
                    f"   Please check your GPU machine and run:\n"
                    f"   python3 server.py --host 0.0.0.0 --port 8000 --model large-v3-turbo --device cuda\n"
                    f"   (If server.py was just started, allow it ~1-2 min to finish downloading model weights)"
                )
            else:
                logger.error(f"Could not reach remote GPU server at {self.server_url}: {exc}")

    def _resolve_device_and_compute(self) -> Tuple[str, str]:
        """Validate device and compute type, with automatic fallback for CPU and CUDA."""
        device = self.requested_device.lower()
        compute = self.requested_compute_type.lower()

        # Resolve auto device
        if device == "auto":
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"

        # Validate compute type on CPU
        if device == "cpu":
            if compute in ("float16", "bfloat16"):
                logger.warning(
                    f"Compute type '{compute}' is typically not supported on CPU. "
                    f"Falling back to 'int8' for optimal performance."
                )
                compute = "int8"
            elif compute not in ("int8", "int8_float32", "int16", "float32"):
                compute = "int8"

        return device, compute

    def _load_model(self) -> None:
        """Instantiate faster_whisper WhisperModel with error handling and fallback."""
        self.resolved_device, self.resolved_compute_type = self._resolve_device_and_compute()
        logger.info(
            f"Loading faster-whisper model '{self.model_size}' "
            f"[device={self.resolved_device}, compute_type={self.resolved_compute_type}]..."
        )

        from faster_whisper import WhisperModel

        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.resolved_device,
                compute_type=self.resolved_compute_type,
            )
            logger.info("faster-whisper model loaded successfully.")
        except Exception as exc:
            # If failed on CUDA or specific compute type, fallback to CPU int8
            if self.resolved_device == "cuda" or self.resolved_compute_type != "int8":
                logger.warning(
                    f"Failed to load model with {self.resolved_device}/{self.resolved_compute_type}: {exc}. "
                    "Retrying with cpu/int8 fallback..."
                )
                self.resolved_device = "cpu"
                self.resolved_compute_type = "int8"
                self.model = WhisperModel(
                    self.model_size,
                    device=self.resolved_device,
                    compute_type=self.resolved_compute_type,
                )
                logger.info("faster-whisper model loaded with cpu/int8 fallback.")
            else:
                raise exc

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("Transcriber thread started.")
        while not self._stop_event.is_set():
            try:
                segment = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._transcribe_segment(segment)
            except Exception as exc:
                logger.error(f"Error during transcription inference: {exc}", exc_info=True)
            finally:
                self.audio_queue.task_done()

    def _transcribe_segment(self, segment: AudioSegment) -> None:
        """Run language identification and transcription on the speech segment."""
        if self.server_url:
            self._handle_remote_server_transcription(segment)
            return

        if self.mock_mode or self.model is None:
            self._handle_mock_transcription(segment)
            return

        audio_data = segment.data

        # Detect spoken language restricted strictly to whitelist
        target_lang, lang_prob = self._identify_whitelisted_language(audio_data)

        # Transcribe with the detected target language
        # beam_size=1, best_of=1, temperature=0.0 gives lowest latency
        segments, info = self.model.transcribe(
            audio_data,
            language=target_lang,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,  # Already filtered by our low-latency Silero VAD
            without_timestamps=True,
        )

        full_text = " ".join([s.text for s in segments])
        cleaned_text = clean_transcribed_text(full_text)

        if not cleaned_text:
            logger.debug("Discarded empty or hallucinated transcription segment.")
            return

        final_lang = target_lang or info.language
        confidence = lang_prob if lang_prob > 0 else info.language_probability

        result = TranscriptionResult(
            text=cleaned_text,
            language=final_lang,
            confidence=confidence,
            timestamp=segment.timestamp,
            duration_s=segment.duration_s,
        )

        logger.info(
            f"Transcribed [{final_lang.upper()}] ({confidence*100:.0f}%): {cleaned_text}"
        )
        self.result_callback(result)

    def _handle_remote_server_transcription(self, segment: AudioSegment) -> None:
        """Offload audio chunk to remote GPU server for ultra-low latency."""
        import base64
        import requests
        from translator import TranslationBundle

        raw_b64 = base64.b64encode(segment.data.tobytes()).decode("utf-8")
        payload = {
            "audio_base64": raw_b64,
            "targets": self.target_languages,
            "whitelist": list(self.whitelist) if self.whitelist else None,
            "sample_rate": segment.sample_rate,
        }
        try:
            r = requests.post(f"{self.server_url}/process", json=payload, timeout=6.0)
            if r.status_code == 200:
                data = r.json()
                orig_text = data.get("original_text", "")
                if not orig_text:
                    return

                spoken_lang = data.get("source_lang", "en")
                translations = data.get("translations", {})
                latency_ms = data.get("latency_ms", 0.0)

                logger.info(f"GPU [{spoken_lang.upper()}] ({latency_ms:.0f}ms): {orig_text}")

                if self.bundle_callback and translations:
                    bundle = TranslationBundle(
                        source_lang=spoken_lang,
                        original_text=orig_text,
                        translations=translations,
                        timestamp=segment.timestamp,
                        latency_ms=latency_ms,
                    )
                    self.bundle_callback(bundle)
                else:
                    result = TranscriptionResult(
                        text=orig_text,
                        language=spoken_lang,
                        confidence=data.get("confidence", 1.0),
                        timestamp=segment.timestamp,
                        duration_s=segment.duration_s,
                    )
                    self.result_callback(result)
            else:
                logger.warning(f"Remote GPU server returned status {r.status_code}: {r.text}")
        except Exception as exc:
            now = time.time()
            # Throttle repetitive connection error messages to once every 10 seconds
            if now - self._last_server_error_log > 10.0:
                self._last_server_error_log = now
                err_str = str(exc)
                if "Connection refused" in err_str:
                    logger.error(
                        f"Remote GPU server ({self.server_url}) connection refused. "
                        "Is server.py running on your GPU host?"
                    )
                else:
                    logger.error(f"Failed to communicate with remote GPU server: {exc}")

    def _identify_whitelisted_language(
        self, audio: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """Identify spoken language restricted strictly to the configured whitelist.

        Returns:
            Tuple of (language_code, probability). If no whitelist is specified,
            returns (None, 0.0) to allow Whisper to auto-detect across all languages.
        """
        if not self.whitelist:
            # No restriction specified: Whisper auto-detects naturally
            return None, 0.0

        try:
            # Faster-whisper feature extraction
            features = self.model.feature_extractor(audio)
            # Pad/trim to 3000 frames (30s) as expected by Whisper encoder
            import ctranslate2

            segment_features = np.zeros((features.shape[0], 3000), dtype=np.float32)
            usable_frames = min(features.shape[1], 3000)
            segment_features[:, :usable_frames] = features[:, :usable_frames]

            # Convert to CTranslate2 StorageView
            storage = ctranslate2.StorageView.from_array(
                np.expand_dims(segment_features, axis=0)
            )

            # Query language probability distribution
            encoder_output = self.model.model.encode(storage)
            results = self.model.model.detect_language(encoder_output)

            # Results is a list of [(lang_token, prob), ...]
            # Filter strictly to whitelisted languages
            whitelisted_candidates = []
            for item in results[0]:
                token, prob = item
                # Token format is e.g. '<|en|>' or 'en'
                lang_code = token.strip("<|>").lower()
                if lang_code in self.whitelist:
                    whitelisted_candidates.append((lang_code, prob))

            if whitelisted_candidates:
                # Select the highest probability language among the whitelist
                whitelisted_candidates.sort(key=lambda x: x[1], reverse=True)
                best_lang, best_prob = whitelisted_candidates[0]
                logger.debug(
                    f"Whitelist language ID: selected '{best_lang}' (p={best_prob:.3f}) "
                    f"from candidates: {whitelisted_candidates}"
                )
                return best_lang, best_prob

        except Exception as exc:
            logger.debug(f"Direct language identification head error: {exc}")

        # Fallback: Pick first language from whitelist or let Whisper detect
        default_lang = next(iter(self.whitelist)) if self.whitelist else None
        return default_lang, 0.5

    def _handle_mock_transcription(self, segment: AudioSegment) -> None:
        """Simulated transcription for testing environments."""
        sample_phrases = [
            ("Welcome everyone to today's live demonstration.", "en"),
            ("We are showing real-time multi-language subtitles.", "en"),
            ("The architecture delivers ultra-low latency.", "en"),
            ("El sistema traduce automáticamente en tiempo real.", "es"),
            ("很高兴向大家展示实时多语言字幕系统。", "zh"),
            ("Das System arbeitet vollkommen lokal und plattformübergreifend.", "de"),
        ]
        import random

        phrase, lang = random.choice(sample_phrases)
        if self.whitelist and lang not in self.whitelist:
            lang = next(iter(self.whitelist))

        result = TranscriptionResult(
            text=phrase,
            language=lang,
            confidence=0.98,
            timestamp=segment.timestamp,
            duration_s=segment.duration_s,
        )
        time.sleep(0.1)
        self.result_callback(result)
