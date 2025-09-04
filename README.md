# One-Stop Gen-AI Platform

This project integrates **AI-assisted code debugging** and a **cinematic content generation pipeline** into a single FastAPI + Gradio application.

## 🚀 Features
- **Code Analysis**
  - Syntax & semantic checks
  - Runtime error detection
  - Dependency version checks
  - Automatic fixes via LLM (Ollama WizardCoder)

- **Cinematic Pipeline**
  - Text-to-video generation (Zeroscope / Diffusers)
  - Script writing (DistilGPT2)
  - Text-to-speech (Bark / Coqui TTS fallback)
  - Lip-sync (Wav2Lip)
  - Emotion animation (SadTalker)
  - Background music generation (Riffusion)
  - Final video stitching with audio/music

- **Other Tools**
  - Text generation & paraphrasing
  - Code generation (StarCoder, CodeGen)
  - Text-to-Image (Stable Diffusion / AUTOMATIC1111 API)
  - Image-to-Image editing
  - 3D object generation (DreamFusion)

## 🛠️ Tech Stack
- **Backend:** FastAPI, Uvicorn
- **UI:** Gradio
- **ML/GenAI:** Hugging Face Transformers, Diffusers, Torch, MoviePy, Riffusion, Wav2Lip, SadTalker
- **External APIs:** Ollama, AUTOMATIC1111 SD WebUI, Oobabooga text-generation-webui

## 📦 Installation
```bash
# Clone repo
git clone https://github.com/<your-username>/one-stop-genai.git
cd one-stop-genai

# Create venv
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
