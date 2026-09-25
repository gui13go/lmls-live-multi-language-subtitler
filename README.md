# Live Multi-Language Subtitles Overlay 🌐🎙️

A high-performance, cross-platform (Linux Ubuntu and Windows 10/11) desktop overlay application that continuously captures microphone input, identifies spoken language dynamically, and displays live subtitles across $N$ target languages in a floating bar on top of all active windows.

Designed specifically for live presentations, conferences, slides, and lectures where the presenter needs real-time translation without interrupting mouse clicks or slide changes.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Microphone Input (16 kHz)       |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         sounddevice Stream            |
                      |  - 512-sample (32ms) audio chunks     |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       Silero VAD + Ring Buffer        |
                      |  - Pre-speech padding (250ms)         |
                      |  - Sliding window (1.5s - 2.5s)       |
                      |  - Silence timeout & dead-air filter  |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      faster-whisper (CTranslate2)     |
                      |  - Automatic Language ID (Whitelist)  |
                      |  - Quantized inference (int8/float16) |
                      |  - Beam size 1 / greedy (low latency) |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     Decoupled Translation Worker      |
                      |  - If Target == Spoken: Passthrough   |
                      |  - If Target != Spoken: Parallel API  |
                      |  - ThreadPoolExecutor + LRU cache     |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         PyQt6 Floating Overlay        |
                      |  - Frameless, translucent background  |
                      |  - AlwaysOnTop                        |
                      |  - Mouse Click-Through Pass-Through   |
                      |  - Global Hotkeys (Ctrl+Shift+H/T/Q)  |
                      +---------------------------------------+
```

---

## Key Features

1. **Ultra-Low Latency Speech Pipeline**:
   - Continuous 16 kHz mono capture via `sounddevice`.
   - Real-time Voice Activity Detection using `silero-vad` (ONNX / Torch) on 32ms frames.
   - Rolling ring buffer (1.5s–2.5s sliding window) with pre-speech padding to prevent clipping the first syllable.
   - Silence detector automatically filters out dead air and ambient background noise.

2. **Dynamic Spoken Language Identification**:
   - Uses Whisper's encoder language head restricted strictly to the user's `--whitelist` (e.g., `--whitelist en,zh,de`).
   - Automatically detects language switches between whitelisted speakers on the fly.

3. **Decoupled Multi-Language Translation Layer**:
   - Zero-latency passthrough if the spoken language matches a target language.
   - Parallel concurrent execution via `ThreadPoolExecutor` across arbitrary $N$ target languages (`--targets en,zh,de,es,fr`).
   - Multi-engine support via `deep-translator` (Google Translate web engine, DeepL, or MyMemory) with in-memory LRU caching.

4. **Floating Frameless Click-Through HUD**:
   - Frameless, semi-transparent frosted card (`rgba(12, 16, 24, 0.72)`) with rounded corners.
   - **True Mouse Click-Through**: Clicks pass straight through to underlying slide decks, Google Slides, PowerPoint, or video players (`WA_TransparentForMouseEvents`, Windows Win32 `WS_EX_TRANSPARENT`, Linux X11 shape masks).
   - High-contrast, scalable typography (`--font-size`) with drop shadows and distinct language pill badges.
   - **Movable Mode**: Press `Ctrl+Shift+T` to temporarily disable click-through and drag the floating bar to any desired location on your monitor.

5. **Hotkeys**:
   - `Ctrl+Shift+H`: Toggle overlay visibility (Hide / Show).
   - `Ctrl+Shift+T`: Toggle Click-Through vs Movable / Draggable mode.
   - `Ctrl+Shift+C`: Clear current subtitles immediately.
   - `Ctrl+Shift+Q`: Clean exit.

---

## File Structure

- **`audio.py`**: Audio streaming input via `sounddevice`, rolling ring buffer, pre-speech padding, and `silero-vad` filtering.
- **`transcriber.py`**: `faster-whisper` model lifecycle, whitelisted language ID, greedy low-latency transcription, and hallucination sanitizer.
- **`translator.py`**: Decoupled parallel translation worker, direct language passthrough, and LRU cache.
- **`gui.py`**: Frameless always-on-top transparent PyQt6 overlay, Win32/X11 click-through setup, responsive layout, and global hotkeys.
- **`main.py`**: CLI parser, pipeline orchestrator, signal handling, and clean shutdown.
- **`requirements.txt`**: Pinned, verified dependencies.

---

## Installation & Setup

### Prerequisites

- **Python**: 3.10 – 3.14
- **Microphone**: Working audio input device.

### 1. Ubuntu (Linux 20.04 / 22.04 / 24.04+)

Install system dependencies for audio capture and Qt6:
```bash
sudo apt update
sudo apt install -y portaudio19-dev libxcb-cursor0 libxkbcommon-x11-0
```

Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You can now run directly with:
```bash
./run.sh --targets en,zh,de --whitelist en,es,zh
# or
python3 main.py --targets en,zh,de --whitelist en,es,zh
```
*(Note: `main.py` automatically detects and delegates to `.venv` if run with system `python3`)*

> **Note for Wayland Users**: If running under a Wayland desktop (e.g., GNOME Wayland), you can run with `QT_QPA_PLATFORM=xcb` for full X11 compatibility or run natively on Wayland:
> ```bash
> QT_QPA_PLATFORM=xcb python main.py
> ```

### 2. Windows (10 / 11)

1. Open PowerShell or Command Prompt.
2. Create and activate a virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## CLI Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--server` | `None` | Remote GPU server URL (e.g., `http://192.168.1.100:8000`) for ultra-low latency inference. |
| `--model` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3-turbo`). |
| `--compute-type` | `int8` | Quantization type (`int8`, `float16`, `bfloat16`, `float32`). |
| `--device` | `auto` | Execution device (`auto`, `cpu`, `cuda`). |
| `--targets` | `en,zh,de` | Comma-separated target languages to display simultaneously. |
| `--whitelist` | `en,zh,de` | Allowed spoken language codes for automatic language ID. |
| `--position` | `top` | Screen position (`top` or `bottom`). |
| `--font-size` | `20` | Subtitle font size in pixels. |
| `--max-speech` | `1.8` | Continuous speech window in seconds (lower values = faster subtitles). |
| `--silence-timeout`| `400` | Silence duration in ms to trigger end of utterance (lower = faster response). |
| `--start-locked` | `False` | Start with click-through locked immediately (default is unlocked/movable). |
| `--opacity` | `0.72` | Background overlay opacity (`0.1` to `1.0`). |
| `--vad-threshold`| `0.5` | Voice activity detection sensitivity (`0.1` to `0.9`). |
| `--mic-index` | `None` | Audio device index (run `--list-devices` to view). |
| `--list-devices` | - | List all available input microphones and exit. |
| `--engine` | `google` | Translation engine (`google`, `deepl`, `mymemory`). |
| `--deepl-key` | `None` | DeepL API Key (if `--engine deepl` is selected). |
| `--demo` | - | Runs in synthetic audio mode (demo without mic). |
| `--mock-stt` | - | Simulated transcription without downloading Whisper weights. |

---

## ⚡ How to Make It Run Much Faster

### Option A: Local CPU Optimization (No GPU Server Needed)
If running entirely on your laptop CPU:
1. **Use `tiny` or `base` instead of `small`**: `small` requires ~2.0s on laptop CPUs, whereas `base` takes ~0.4s and `tiny` takes ~0.15s (over 10x faster)!
2. **Shorten the speech buffer**: By default `--max-speech` is `1.8s` and `--silence-timeout` is `400ms`.
```bash
python3 main.py --model base --compute-type int8 --max-speech 1.4 --silence-timeout 350 --targets en,zh,de --whitelist en,es,zh
```

### Option B: Remote GPU Server Offloading (Ultra-Fast ~40ms Latency)
Run the heavy neural network on your remote machine that has NVIDIA GPUs, while streaming microphone audio directly from your local laptop:

1. **On your Remote GPU Server** (SSH into your GPU machine):
```bash
# Clone or pull the repository:
git pull

# Run the automated server script (automatically creates .venv & installs GPU packages):
./run_server.sh --host 0.0.0.0 --port 8000 --model large-v3-turbo --device cuda --compute-type float16
```
*(Or manually: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-server.txt && python3 server.py --host 0.0.0.0 --port 8000 --model large-v3-turbo --device cuda --compute-type float16`)*

*(Tip: On NVIDIA GPUs, `float16` or `bfloat16` with `large-v3-turbo` achieves state-of-the-art accuracy in ~30–50ms!)*

2. **On your Local Laptop / Client**:
```bash
python3 main.py --server http://<YOUR_GPU_SERVER_IP>:8000 --targets en,zh,de --whitelist en,es,zh
```
Your laptop captures microphone audio and sends it over the local network / LAN / VPN to the server; the server runs the GPU inference and sends translations back to your floating subtitle overlay instantly!

---

## Execution Examples

### 1. Standard Live Presentation (Top of Screen)
Transcribe speech dynamically (English, Spanish, or Chinese) and output subtitles in English, Chinese, and German:
```bash
python main.py --targets en,zh,de --whitelist en,es,zh --position top --font-size 22
```

### 2. Bottom Floating Bar with Larger Font (e.g., for Lectures)
```bash
python main.py --targets en,es,fr,ja --whitelist en,es --position bottom --font-size 26
```

### 3. Ultra-Fast CPU Execution (Low Latency)
Using the `base` or `small` model with `int8` quantization:
```bash
python main.py --model base --compute-type int8 --device cpu --targets en,zh,de
```

### 4. High-Accuracy GPU Mode (NVIDIA CUDA)
```bash
python main.py --model large-v3-turbo --compute-type float16 --device cuda --targets en,zh,de,ja,fr
```

### 5. Selecting a Specific Microphone
First, list connected input devices:
```bash
python main.py --list-devices
```
Then run with the selected microphone index:
```bash
python main.py --mic-index 2 --targets en,zh
```

### 6. Demo / Test Mode (No Physical Mic Required)
Generates synthetic speech audio and runs the full STT, translation, and overlay GUI:
```bash
python main.py --demo --targets en,zh,de,es
```

---

## Interactive Drag & Drop & Click-Through Controls

- **Drag and Drop Repositioning**:
  By default, the overlay opens in **Movable Mode** (`[ 🔓 Movable: ON ]` and `[ ⠿ DRAG TO MOVE ]`).
  Simply **click and drag anywhere** on the top bar or window background to position the overlay anywhere on your screen (top, bottom, center, dual monitors).
- **Locking Click-Through**:
  Once positioned, click the **`[ 🔓 Movable: ON (Click to Lock) ]`** button or press **`Ctrl+Shift+T`** to lock the overlay into **Pass-Through Mode** (`[ 🔒 Pass-Through: ON ]`).
  While locked, all mouse clicks pass completely through to your underlying slides or presentation!
- **Unlocking / Repositioning Anytime**:
  Press **`Ctrl+Shift+T`** anytime to unlock and drag the overlay to a new location.
- **Hiding / Showing**:
  Press **`Ctrl+Shift+H`** to instantly hide or reveal the subtitle overlay during your presentation.
- **Clearing Subtitles**:
  Press **`Ctrl+Shift+C`** (or click the **Clear** button on the header) to wipe the subtitles immediately.
- **Quitting**:
  Press **`Ctrl+Shift+Q`** or `Ctrl+C` in the terminal to cleanly exit.
