"""audio.py - Microphone audio capture, rolling ring buffer, and Voice Activity Detection (VAD).

Continuously streams 16 kHz mono audio via sounddevice in an isolated background thread,
maintains a rolling ring buffer (1.5s - 2.5s sliding window), and filters speech segments
using Silero VAD to avoid transcribing dead air.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("live_subtitles.audio")

# Standard Whisper audio parameters
SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512  # 512 samples @ 16kHz = 32ms (native Silero VAD chunk size)


def _ensure_portaudio_loaded() -> None:
    """Ensure libportaudio is discoverable by sounddevice even without system-wide install."""
    import ctypes.util
    import os
    import sys

    # If already findable, do nothing
    if ctypes.util.find_library("portaudio"):
        return

    # Check local workspace lib/ directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "lib", "libportaudio.so.2"),
        os.path.join(base_dir, "lib", "libportaudio.so"),
        os.path.join(base_dir, ".venv", "lib", "libportaudio.so.2"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            _orig_find_library = ctypes.util.find_library

            def _custom_find_library(name: str):
                if name in ("portaudio", "libportaudio"):
                    return candidate
                return _orig_find_library(name)

            ctypes.util.find_library = _custom_find_library
            logger.debug(f"Configured local PortAudio fallback: {candidate}")
            break


_ensure_portaudio_loaded()


@dataclass
class AudioSegment:
    """Represents a discrete speech segment ready for transcription."""

    data: np.ndarray  # float32 1D numpy array normalized to [-1.0, 1.0]
    sample_rate: int
    timestamp: float
    duration_s: float


def list_audio_devices() -> List[Dict[str, Any]]:
    """List all available audio input devices on the system.

    Returns:
        List of dicts containing device index, name, hostapi, and max input channels.
    """
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devs = []
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                input_devs.append(
                    {
                        "index": idx,
                        "name": dev.get("name", "Unknown"),
                        "hostapi": dev.get("hostapi", -1),
                        "max_channels": dev.get("max_input_channels", 1),
                        "default_samplerate": dev.get("default_samplerate", 16000),
                    }
                )
        return input_devs
    except Exception as exc:
        logger.error(f"Failed to query audio devices: {exc}")
        return []


class SileroVADDetector:
    """Wrapper around Silero VAD model with fallback mechanisms."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self.is_onnx = False
        self._init_model()

    def _init_model(self) -> None:
        """Attempt loading Silero VAD via ONNX runtime first, then PyTorch, then fallback."""
        try:
            from silero_vad import load_silero_vad

            # Try ONNX first for faster/lighter inference without Torch overhead
            try:
                self.model = load_silero_vad(onnx=True)
                self.is_onnx = True
                logger.info("Loaded Silero VAD with ONNX backend.")
                return
            except Exception as e_onnx:
                logger.debug(f"ONNX Silero VAD load failed ({e_onnx}), trying PyTorch...")

            # Fallback to PyTorch backend
            self.model = load_silero_vad(onnx=False)
            self.is_onnx = False
            logger.info("Loaded Silero VAD with PyTorch backend.")
        except Exception as exc:
            logger.warning(
                f"Could not load silero-vad model ({exc}). Falling back to energy/RMS VAD."
            )
            self.model = None

    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """Compute speech probability for a 512-sample float32 chunk.

        Args:
            chunk: 1D numpy array of 512 float32 samples at 16kHz.

        Returns:
            Probability value in range [0.0, 1.0].
        """
        if self.model is not None:
            try:
                if self.is_onnx:
                    # Silero ONNX accepts 1D float32 or torch tensor
                    prob = float(self.model(chunk, SAMPLE_RATE))
                    return prob
                else:
                    import torch

                    tensor = torch.from_numpy(chunk).float()
                    prob = float(self.model(tensor, SAMPLE_RATE))
                    return prob
            except Exception as exc:
                logger.debug(f"Silero VAD inference error: {exc}")

        # Fallback: Root Mean Square (RMS) energy thresholding
        rms = np.sqrt(np.mean(chunk**2) + 1e-12)
        # Typical silence is < 0.01; speech is > 0.03
        prob = min(1.0, max(0.0, (rms - 0.008) / 0.04))
        return float(prob)

    def is_speech(self, chunk: np.ndarray) -> Tuple[bool, float]:
        prob = self.get_speech_prob(chunk)
        return (prob >= self.threshold, prob)


class AudioCapture(threading.Thread):
    """Continuously captures audio from microphone, processes VAD in real-time,

    and emits speech segments through an output queue.
    """

    def __init__(
        self,
        output_queue: queue.Queue[AudioSegment],
        device_index: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = VAD_CHUNK_SIZE,
        vad_threshold: float = 0.5,
        pre_speech_ms: int = 250,
        silence_timeout_ms: int = 600,
        max_speech_s: float = 2.5,
        min_speech_ms: int = 250,
        simulated_input: bool = False,
    ):
        super().__init__(daemon=True, name="AudioCaptureThread")
        self.output_queue = output_queue
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad_threshold = vad_threshold
        self.simulated_input = simulated_input

        # Conversions to frame counts
        chunk_duration_ms = (chunk_size / sample_rate) * 1000.0  # 32ms
        self.pre_speech_chunks = max(1, int(pre_speech_ms / chunk_duration_ms))
        self.silence_timeout_chunks = max(1, int(silence_timeout_ms / chunk_duration_ms))
        self.max_speech_chunks = int((max_speech_s * 1000.0) / chunk_duration_ms)
        self.min_speech_chunks = max(1, int(min_speech_ms / chunk_duration_ms))

        # Circular buffer for pre-speech frames
        self.pre_speech_ring = collections.deque(maxlen=self.pre_speech_chunks)

        # Queues and lifecycle
        self._raw_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=300)
        self._stop_event = threading.Event()
        self.vad = SileroVADDetector(threshold=vad_threshold)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Sounddevice streaming callback."""
        if status:
            logger.debug(f"Audio stream status: {status}")
        if self._stop_event.is_set():
            return
        # Ensure 1D float32 mono array
        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        try:
            self._raw_queue.put_nowait(chunk)
        except queue.Full:
            # Drop frame if processing is severely lagging to prevent memory blowup
            pass

    def stop(self) -> None:
        """Signal thread to stop and clean up resources."""
        self._stop_event.set()

    def run(self) -> None:
        """Main processing loop."""
        logger.info(
            f"Starting AudioCapture thread (device={self.device_index}, "
            f"simulated={self.simulated_input})."
        )

        if self.simulated_input:
            self._run_simulated_loop()
            return

        import sounddevice as sd

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                device=self.device_index,
                callback=self._audio_callback,
            ):
                logger.info("Microphone stream opened successfully.")
                self._process_audio_loop()
        except Exception as exc:
            logger.error(f"Failed to open microphone stream: {exc}")
            logger.info("Falling back to simulated audio input mode.")
            self._run_simulated_loop()

    def _process_audio_loop(self) -> None:
        """Consumes raw audio chunks, tracks speech state, and builds speech segments."""
        active_speech_chunks: List[np.ndarray] = []
        is_speaking = False
        consecutive_silence_chunks = 0

        while not self._stop_event.is_set():
            try:
                chunk = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            speech_detected, prob = self.vad.is_speech(chunk)

            if speech_detected:
                if not is_speaking:
                    # Speech began! Prepend pre-speech frames
                    is_speaking = True
                    active_speech_chunks = list(self.pre_speech_ring)
                    logger.debug("Speech activity detected. Buffering started.")
                active_speech_chunks.append(chunk)
                consecutive_silence_chunks = 0
            else:
                if is_speaking:
                    consecutive_silence_chunks += 1
                    active_speech_chunks.append(chunk)

                    # Check if silence timeout reached or max speech duration hit
                    if (
                        consecutive_silence_chunks >= self.silence_timeout_chunks
                        or len(active_speech_chunks) >= self.max_speech_chunks
                    ):
                        self._emit_segment(active_speech_chunks, consecutive_silence_chunks)
                        # Reset speech state
                        is_speaking = False
                        active_speech_chunks = []
                        consecutive_silence_chunks = 0
                else:
                    # Dead air: simply roll pre-speech buffer
                    self.pre_speech_ring.append(chunk)

            # Continuous speech limit safeguard: if someone talks continuously for > max_speech_s
            if is_speaking and len(active_speech_chunks) >= self.max_speech_chunks:
                self._emit_segment(active_speech_chunks, 0)
                # Keep last 150ms (~5 chunks) as overlap for speech continuity
                overlap_chunks = min(5, len(active_speech_chunks))
                active_speech_chunks = active_speech_chunks[-overlap_chunks:]
                consecutive_silence_chunks = 0

    def _emit_segment(
        self, chunks: List[np.ndarray], trailing_silence_chunks: int
    ) -> None:
        """Combine accumulated chunks into an AudioSegment and push to output queue."""
        # Trim trailing silence chunks if present
        if trailing_silence_chunks > 0 and len(chunks) > trailing_silence_chunks:
            usable_chunks = chunks[:-trailing_silence_chunks]
        else:
            usable_chunks = chunks

        if len(usable_chunks) < self.min_speech_chunks:
            # Segment too brief, likely a click or cough
            return

        combined_data = np.concatenate(usable_chunks, axis=0)
        # Normalize and clip
        combined_data = np.clip(combined_data, -1.0, 1.0)
        duration_s = len(combined_data) / self.sample_rate

        segment = AudioSegment(
            data=combined_data,
            sample_rate=self.sample_rate,
            timestamp=time.time(),
            duration_s=duration_s,
        )
        try:
            self.output_queue.put_nowait(segment)
            logger.debug(
                f"Dispatched speech segment: {duration_s:.2f}s "
                f"({len(combined_data)} samples)."
            )
        except queue.Full:
            logger.warning("Transcription queue full. Dropped audio segment.")

    def _run_simulated_loop(self) -> None:
        """Generates synthetic audio bursts for demonstration/testing."""
        logger.info("Running in synthetic audio mode (demo / automated test).")
        time.sleep(1.0)
        sample_rate = self.sample_rate

        test_sentences_duration = [2.0, 1.8, 2.2]
        idx = 0

        while not self._stop_event.is_set():
            dur = test_sentences_duration[idx % len(test_sentences_duration)]
            idx += 1
            t = np.linspace(0, dur, int(sample_rate * dur), endpoint=False, dtype=np.float32)
            # Create synthetic speech-like multi-frequency waveform
            audio = (
                0.3 * np.sin(2 * np.pi * 220 * t)
                + 0.2 * np.sin(2 * np.pi * 440 * t)
                + 0.1 * np.random.normal(0, 0.05, len(t)).astype(np.float32)
            )
            audio = np.clip(audio, -1.0, 1.0)

            segment = AudioSegment(
                data=audio,
                sample_rate=sample_rate,
                timestamp=time.time(),
                duration_s=dur,
            )
            try:
                self.output_queue.put(segment, timeout=1.0)
            except queue.Full:
                pass

            # Pause between synthetic phrases
            for _ in range(40):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)
