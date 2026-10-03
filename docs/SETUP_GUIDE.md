# Zero-Cost Local Environment Setup Guide

Follow this guide to set up the required external dependencies and Python virtual environment.

---

## 1. Python Version Selection (Crucial)

Your system has:
- `Python 3.14` (default in PATH)
- `Python 3.11` (installed at `C:\Users\sneha\AppData\Local\Programs\Python\Python311\python.exe`)

> **IMPORTANT**: AI, speech, and audio packages (`torch`, `onnxruntime`, `faster-whisper`, `sounddevice`) currently do **not** have prebuilt binaries for Python 3.14 on Windows.
> You **must** create your virtual environment using **Python 3.11**.

---

## 2. Install Missing System Tools (Manual)

### A. FFmpeg (Audio decoding / conversion)
Open PowerShell (as Administrator or regular user) and run:
```powershell
winget install Gyan.FFmpeg
```
*Or download the zip from https://www.gyan.dev/ffmpeg/builds/ and add the `bin` directory to your system PATH.*

To verify:
```powershell
ffmpeg -version
```

### B. Ollama (Local Zero-Cost LLM Engine)
Install Ollama via winget or installer:
```powershell
winget install Ollama.Ollama
```
*Or download directly from https://ollama.com/download/windows.*

After installation, start Ollama and pull the recommended lightweight model:
```powershell
ollama pull qwen2.5:1.5b
```
*(The 1.5B model runs smoothly on CPU and supports native tool/function calling).*

---

## 3. Create Python 3.11 Virtual Environment

From the root project directory (`d:\Prism`):

```powershell
# 1. Create virtual environment targeting Python 3.11
py -3.11 -m venv .venv

# 2. Activate virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1

# (If PowerShell blocks script execution, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# 3. Verify the active Python version in the venv
python --version
# Should output: Python 3.11.x

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## 4. Setup Environment File

Copy the template:
```powershell
Copy-Item .env.example .env
```

---

## 5. Verify Local Voice Pipeline (Stage 4)

Test the complete test suite (deterministic, offline):
```powershell
pytest -q
```

Verify audio devices on Windows:
```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Verify local ONNX VAD:
```powershell
python -c "from src.vad import SileroVAD; vad = SileroVAD(); print('Silero ONNX VAD ready!')"
```
