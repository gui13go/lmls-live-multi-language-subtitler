"""main.py - CLI entry point and pipeline orchestrator for live multi-language subtitle overlay.

Connects continuous microphone capture (sounddevice + Silero VAD) ->
Whisper STT (faster-whisper with whitelisted language ID) ->
Parallel Translation Worker ->
Frameless click-through Qt6 desktop overlay.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import sys
from typing import List, Optional

# Automatically re-execute within local .venv if executed with system python
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    _curr_dir = os.path.dirname(os.path.abspath(__file__))
    _venv_py = (
        os.path.join(_curr_dir, ".venv", "Scripts", "python.exe")
        if sys.platform == "win32"
        else os.path.join(_curr_dir, ".venv", "bin", "python")
    )
    if os.path.isfile(_venv_py) and os.path.abspath(sys.executable) != os.path.abspath(_venv_py):
        _args = sys.orig_argv[1:] if hasattr(sys, "orig_argv") else sys.argv
        os.execv(_venv_py, [_venv_py] + _args)

# Force X11/XCB backend on Linux desktops when DISPLAY is available.
# This ensures frameless overlays have full window manager authority
# to position anywhere on screen, support click-through transparency,
# stay on top, and register global hotkeys.
if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ and os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication

from audio import AudioCapture, AudioSegment, list_audio_devices
from gui import SubtitleOverlay, setup_global_hotkeys
from transcriber import TranscriptionResult, WhisperTranscriber
from translator import TranslationBundle, TranslationWorker


def setup_logging(level_name: str = "INFO") -> None:
    """Configure structured logging format."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments according to core specifications."""
    parser = argparse.ArgumentParser(
        description="Cross-platform live multi-language subtitle overlay desktop application."
    )

    # Core Requirements
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="Whisper model size/variant (e.g., tiny, base, small, medium, large-v3-turbo; default: small).",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Quantization type (int8, float16, bfloat16, float32; default: int8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Target execution device (cpu, cuda, auto; default: auto).",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="en,zh,de",
        help="Comma-separated list of target language codes to display (e.g., --targets en,zh,de).",
    )
    parser.add_argument(
        "--whitelist",
        type=str,
        default="en,zh,de",
        help="Allowed spoken language codes for automatic language identification (e.g., --whitelist en,zh,de).",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="top",
        choices=["top", "bottom"],
        help="Screen position (top or bottom; default: top).",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=20,
        help="Subtitle typography size in pixels (default: 20).",
    )

    # Audio & Hardware Options
    parser.add_argument(
        "--mic-index",
        type=int,
        default=None,
        help="Specific audio input device index (run --list-devices to find index).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all detected microphone/audio input devices and exit.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD speech detection threshold between 0.0 and 1.0 (default: 0.5).",
    )

    # Visual & Translation Tuning
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.72,
        help="Overlay background opacity from 0.1 to 1.0 (default: 0.72).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="google",
        choices=["google", "deepl", "mymemory"],
        help="Translation backend engine (default: google).",
    )
    parser.add_argument(
        "--deepl-key",
        type=str,
        default=None,
        help="Optional DeepL API key (if --engine deepl is selected).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Custom overlay width in pixels (e.g., 900). Default is auto-proportional (65%% screen width).",
    )

    # Remote GPU Server Offloading & Latency Tuning
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="Remote GPU server URL (e.g., http://192.168.1.100:8000) to offload Whisper inference and translations to an NVIDIA GPU.",
    )
    parser.add_argument(
        "--max-speech",
        type=float,
        default=1.8,
        help="Maximum continuous speech window in seconds before triggering transcription (default: 1.8). Lower values (1.2-1.8) yield faster live subtitles.",
    )
    parser.add_argument(
        "--silence-timeout",
        type=int,
        default=400,
        help="Milliseconds of silence to commit a speech segment (default: 400ms).",
    )
    parser.add_argument(
        "--start-locked",
        action="store_true",
        help="Start with mouse click-through locked immediately (default: False, allows dragging and positioning first).",
    )

    # Testing & Debugging
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic audio generation (useful for testing without mic).",
    )
    parser.add_argument(
        "--mock-stt",
        action="store_true",
        help="Use simulated transcription engine without loading heavy Whisper weights.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity level.",
    )

    return parser.parse_args()


def display_audio_devices() -> None:
    """Print available audio input devices in a clear table."""
    devices = list_audio_devices()
    print("\n--- Available Audio Input Devices ---")
    if not devices:
        print("No active audio input devices found.")
    else:
        for dev in devices:
            print(
                f"[{dev['index']}] {dev['name']} "
                f"(Channels: {dev['max_channels']}, Default Rate: {dev['default_samplerate']}Hz)"
            )
    print("-------------------------------------\n")


def main() -> None:
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger("live_subtitles.main")

    # If requested, list devices and terminate
    if args.list_devices:
        display_audio_devices()
        sys.exit(0)

    # Parse and validate target languages
    target_langs = [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    if not target_langs:
        logger.error("No target languages specified. Use --targets en,zh,de")
        sys.exit(1)

    # Parse whitelist languages
    whitelist_langs = (
        [w.strip().lower() for w in args.whitelist.split(",") if w.strip()]
        if args.whitelist
        else None
    )

    logger.info("Initializing Live Subtitle Overlay...")
    logger.info(f"Target Languages : {target_langs}")
    logger.info(f"Whitelist Spoken : {whitelist_langs or 'All'}")
    if args.server:
        logger.info(f"Remote GPU Server : {args.server} (Heavy inference offloaded to GPU)")
    else:
        logger.info(f"Whisper Model    : {args.model} ({args.compute_type} on {args.device})")
    logger.info(f"Screen Position  : {args.position} | Font Size: {args.font_size}px")
    logger.info(f"Speech Buffering : max={args.max_speech}s, silence_timeout={args.silence_timeout}ms")

    # 1. Initialize Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("Live Subtitles Overlay")

    # 2. Instantiate GUI Overlay (starts unlocked/movable by default)
    overlay = SubtitleOverlay(
        target_languages=target_langs,
        position=args.position,
        font_size=args.font_size,
        opacity=args.opacity,
        start_locked=args.start_locked,
        width=args.width,
    )

    # 3. Setup Decoupled Translation Worker
    def on_translation_ready(bundle: TranslationBundle) -> None:
        """Forward completed translation bundle to Qt overlay."""
        overlay.signals.update_subtitles.emit(bundle.translations)

    translation_worker = TranslationWorker(
        target_languages=target_langs,
        result_callback=on_translation_ready,
        engine_name=args.engine,
        api_key=args.deepl_key,
    )

    # 4. Setup Speech-to-Text Transcriber
    audio_queue: queue.Queue[AudioSegment] = queue.Queue(maxsize=100)

    def on_transcription_ready(result: TranscriptionResult) -> None:
        """Forward raw speech transcription to parallel translation worker."""
        translation_worker.submit_transcription(result)

    transcriber = WhisperTranscriber(
        audio_queue=audio_queue,
        result_callback=on_transcription_ready,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        whitelist_languages=whitelist_langs,
        target_languages=target_langs,
        server_url=args.server,
        bundle_callback=on_translation_ready,
        mock_mode=args.mock_stt,
    )

    # 5. Setup Audio Capture
    audio_capture = AudioCapture(
        output_queue=audio_queue,
        device_index=args.mic_index,
        vad_threshold=args.vad_threshold,
        silence_timeout_ms=args.silence_timeout,
        max_speech_s=args.max_speech,
        simulated_input=args.demo,
    )

    # 6. Global & Local Hotkeys
    _pynput_listener = setup_global_hotkeys(overlay.signals)

    # Fallback in-app shortcuts (active when overlay has focus)
    QShortcut(QKeySequence("Ctrl+Shift+H"), overlay, overlay.signals.toggle_visibility.emit)
    QShortcut(QKeySequence("Ctrl+Shift+T"), overlay, overlay.signals.toggle_click_through.emit)
    QShortcut(QKeySequence("Ctrl+Shift+C"), overlay, overlay.signals.clear_subtitles.emit)
    QShortcut(QKeySequence("Ctrl+Shift+Q"), overlay, overlay.signals.quit_app.emit)

    # 7. Start Background Processing Threads
    translation_worker.start()
    transcriber.start()
    audio_capture.start()

    # 8. Show Overlay
    overlay.show()

    # 9. Handle POSIX SIGINT (Ctrl+C) smoothly in Qt Event Loop
    def sigint_handler(*_) -> None:
        logger.info("Received interrupt signal. Exiting gracefully...")
        app.quit()

    signal.signal(signal.SIGINT, sigint_handler)
    # Timer allows Python interpreter to process OS signals periodically
    sig_timer = QTimer()
    sig_timer.start(250)
    sig_timer.timeout.connect(lambda: None)

    # 10. Execute Application Loop
    exit_code = app.exec()

    # Cleanup upon exit
    logger.info("Cleaning up background workers...")
    audio_capture.stop()
    transcriber.stop()
    translation_worker.stop()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
