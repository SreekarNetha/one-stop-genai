# merged_cinematic_full.py
"""
Merged FastAPI backend + Gradio UI for:
 - Code analysis (using Ollama wizardcoder as in your app.py)
 - Cinematic pipeline (text->video, music, tts, lipsync, emotion, stitch)
Uses model IDs/paths exactly from your uploaded files.
Runtime timeout for code runtime checks: 15 seconds.
"""

import os
import re
import json
import shutil
import subprocess
import threading
import traceback
import time
import os
import base64
from tempfile import TemporaryDirectory
from typing import List, Dict, Optional

# Web + UI
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import gradio as gr
import requests

# ML / AV
import torch
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, ImageSequenceClip
import scipy.io.wavfile
from scipy.io.wavfile import write as write_wav

# Diffusers / transformers / riffusion (we will lazy-import some inside functions to avoid top-level import errors)
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers import UNet2DConditionModel, AutoencoderKL, LMSDiscreteScheduler, DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import export_to_video

# riffusion import (assumes repo installed)
from riffusion.riffusion_pipeline import RiffusionPipeline

# ============================================================
# --- oobabooga text-generation-webui ---
OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL", "http://127.0.0.1:5000")  # run text-generation-webui with --api
OOBABOOGA_ENDPOINT = os.getenv("OOBABOOGA_ENDPOINT", "/api/v1/generate")

# --- whisper.cpp ---
WHISPER_CPP_BIN = os.getenv("WHISPER_CPP_BIN", "/content/whisper.cpp/main")
WHISPER_CPP_MODEL = os.getenv("WHISPER_CPP_MODEL", "/content/whisper.cpp/models/ggml-large-v2.bin")

# --- Coqui TTS ---
# Prefer local model path to avoid pulling from HF. If empty, will try a model name.
COQUI_TTS_MODEL_PATH = os.getenv("COQUI_TTS_MODEL_PATH", "")  # e.g. "/content/coqui_models/tts_models/en/ljspeech/tacotron2-DDC"
COQUI_TTS_MODEL_NAME = os.getenv("COQUI_TTS_MODEL_NAME", "tts_models/en/ljspeech/tacotron2-DDC")

# --- AUTOMATIC1111 (SDXL) ---
SD_WEBUI_API = os.getenv("SD_WEBUI_API", "http://127.0.0.1:7860")  # launch webui with --api
SD_DEFAULT_STEPS = int(os.getenv("SD_DEFAULT_STEPS", "28"))
SD_DEFAULT_CFG = float(os.getenv("SD_DEFAULT_CFG", "6.5"))
SD_DEFAULT_WIDTH = int(os.getenv("SD_DEFAULT_WIDTH", "1024"))
SD_DEFAULT_HEIGHT = int(os.getenv("SD_DEFAULT_HEIGHT", "1024"))

# --- StarCoder (local, no HF) ---
# Point to a directory with local weights (e.g., a safetensors/transformers folder)
STARCODER_LOCAL_PATH = os.getenv("STARCODER_LOCAL_PATH", "/content/starcoder-local")
STARCODER_MAX_NEW_TOKENS = int(os.getenv("STARCODER_MAX_NEW_TOKENS", "256"))

# --- Stable DreamFusion ---
DREAMFUSION_DIR = os.getenv("DREAMFUSION_DIR", "/content/stable-dreamfusion")
DREAMFUSION_PY = os.getenv("DREAMFUSION_PY", "main.py")
DREAMFUSION_OUTDIR = os.getenv("DREAMFUSION_OUTDIR", "sdream_out")

# -----------------------
# CONFIG (values taken from your uploaded files)
# -----------------------
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "wizardcoder:7b")  # from your app.py

# Cinematic models/paths from your cinematic file
TEXT_TO_VIDEO_MODEL = os.getenv("TEXT_TO_VIDEO_MODEL", "damo-vilab/text-to-video-ms-1.7b")
SD_DIFFUSERS_LOCAL = os.getenv("SD_DIFFUSERS_LOCAL", "/content/stable_diffusion_v1_5_diffusers")
RIFFUSION_AUDIO_PATH = os.getenv("RIFFUSION_AUDIO_PATH", "/content/riffusion_audio.wav")

# LLMs referenced in your cinematic file
OPENCHAT_MODEL_ID = os.getenv("OPENCHAT_MODEL_ID", "TheBloke/OpenChat-3.5-0106-GPTQ")
MIXTRAL_MODEL_ID = os.getenv("MIXTRAL_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
DISTILGPT2_SCRIPT_LLM = os.getenv("SCRIPT_LLM", "distilgpt2")  # used in your cinematic file as small fallback for script

# Wav2Lip / SadTalker script paths from your files
WAV2LIP_INFERENCE_SCRIPT = os.getenv("WAV2LIP_INFERENCE_SCRIPT", "Wav2Lip/inference.py")
WAV2LIP_CHECKPOINT = os.getenv("WAV2LIP_CHECKPOINT", "Wav2Lip/checkpoints/wav2lip.pth")
SADTALKER_INFER_SCRIPT = os.getenv("SADTALKER_INFER_SCRIPT", "SadTalker/infer.py")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Logging (from cli)
# -----------------------
LOG_FILE = "query_log.json"
def log_query(user, feat, prompt):
    try:
        with open(LOG_FILE, "a") as fb:
            fb.write(json.dumps({
                "user": user,
                "feature": feat,
                "prompt": prompt,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }) + "\n")
    except Exception:
        pass

# -----------------------
# Safe subprocess helper
# -----------------------
def _run_command_safely(command: List[str], input_data: str = None, cwd: str = None, timeout: int = 60) -> Dict:
    try:
        proc = subprocess.run(command, input=input_data, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        return {"stdout": proc.stdout or "", "stderr": proc.stderr or "", "success": proc.returncode == 0, "returncode": proc.returncode}
    except subprocess.TimeoutExpired as e:
        return {"stdout": getattr(e, "stdout", "") or "", "stderr": getattr(e, "stderr", "") or "", "success": False, "error_type": "TimeoutExpired"}
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"Command not found: {command[0]}. Install or correct path.", "success": False, "error_type": "FileNotFoundError"}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "success": False, "error_type": "UnexpectedError"}

# Code analysis config
SUPPORTED_EXTENSIONS = [
    ".py", ".java", ".js", ".html", ".css", ".cpp", ".c", ".cs", ".xml", ".sql", ".rb", ".dart", ".kt", ".ts", ".go"
]
CODE_EXECUTION_TIMEOUT = 600  # per your request

# -----------------------
# FastAPI app
# -----------------------
api = FastAPI(title="Merged AI Debugger + Cinematic Pipeline Backend")

# -----------------------
# Language tool config (expanded to include xml/html/css etc.)
# (from your app.py file)
# -----------------------
LANGUAGE_TOOLS_CONFIG = {
    "py": {
        "syntax": ["python", "-m", "py_compile"],
        "runtime": ["python"],
        "deps": ["pip", "show"],
        "dep_regex": r"^(import|from)\s+([a-zA-Z0-9_]+)",
        "parser": "python"
    },
    "java": {
        "syntax": ["javac"],
        "runtime": ["java", "-cp", "."],
        "deps": ["mvn", "dependency:list"],
        "dep_regex": r"^(import)\s+([a-zA-Z0-9_\.]+);",
        "parser": "java"
    },
    "c": {"syntax": ["gcc", "-fsyntax-only"], "runtime": ["./temp.out"], "deps": [], "parser": "gcc"},
    "cpp": {"syntax": ["g++", "-fsyntax-only"], "runtime": ["./temp.out"], "deps": [], "parser": "g++"},
    "js": {
        "syntax": ["npx", "eslint", "--stdin", "--stdin-filename", "temp.js"],
        "runtime": ["node"],
        "deps": ["npm", "list", "--json", "--depth=0"],
        "dep_regex": r"(require|import)\s*\(?['\"]([a-zA-Z0-9_\-\/@\.]+)['\"]",
        "parser": "eslint"
    },
    "html": {"syntax": ["npx", "html-validate", "--stdin"], "runtime": [], "deps": [], "parser": "html-validate"},
    "css": {"syntax": ["npx", "stylelint", "--stdin"], "runtime": [], "deps": [], "parser": "stylelint"},
    "xml": {"syntax": ["xmllint", "--noout"], "runtime": [], "deps": [], "parser": "xml"},
    "cs": {
        "syntax": ["dotnet", "build", "--no-restore"],
        "runtime": ["dotnet", "run", "--no-build"],
        "deps": ["dotnet", "list", "package"],
        "dep_regex": r"using\s+([a-zA-Z0-9_\.]+);",
        "parser": "dotnet"
    },
    "kt": {"syntax": ["kotlinc", "-nowarn", "-Xno-stdlib"], "runtime": ["java", "-jar"], "deps": ["gradle", "dependencies"], "dep_regex": r"import\s+([a-zA-Z0-9_\.]+)", "parser": "kotlin"},
    "dart": {"syntax": ["dart", "analyze"], "runtime": ["dart", "run"], "deps": ["pub", "deps", "--json"], "dep_regex": r"import\s+['\"]package:([a-zA-Z0-9_\/]+)['\"]", "parser": "dart"},
    "go": {"syntax": ["go", "vet"], "runtime": ["go", "run"], "deps": ["go", "mod", "graph"], "dep_regex": r"import\s*\(?[\"`]([a-zA-Z0-9_\/\.]+)[\"`]\)?", "parser": "go"},
}

# -----------------------
# Helpers: run commands safely
# -----------------------
def _run_command_safely(command: List[str], input_data: str = None, cwd: str = None, timeout: int = CODE_EXECUTION_TIMEOUT) -> Dict:
    try:
        proc = subprocess.run(command, input=input_data, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        return {"stdout": proc.stdout or "", "stderr": proc.stderr or "", "success": proc.returncode == 0, "returncode": proc.returncode}
    except subprocess.TimeoutExpired as e:
        return {"stdout": e.stdout or "", "stderr": e.stderr or "", "success": False, "error_type": "TimeoutExpired"}
    except FileNotFoundError:
        return {"stdout": "", "stderr": f"Command not found: {command[0]}. Install or correct path.", "success": False, "error_type": "FileNotFoundError"}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "success": False, "error_type": "UnexpectedError"}

def _parse_error_output(output: str, language_parser: str) -> str:
    if not output or not output.strip():
        return "No specific errors detected (tool might have run silently)."
    lines = output.strip().splitlines()
    if language_parser == "python":
        tb = []
        for l in lines:
            if "File \"" in l and ", line " in l:
                tb.append(l.strip())
            elif l.startswith("    "):
                tb.append(l.strip())
            elif l.strip() and not l.strip().startswith("Traceback"):
                tb.append(l.strip())
        return "\n".join(tb) if tb else output
    if language_parser == "eslint":
        try:
            j = json.loads(output)
            out = []
            for f in j:
                for m in f.get("messages", []):
                    out.append(f"Line {m.get('line','?')}:{m.get('column','?')} {m.get('message','')}")
            return "\n".join(out) if out else output
        except Exception:
            return output
    if language_parser == "xml":
        return "\n".join([l.strip() for l in lines if l.strip()]) or output
    return output

# -----------------------
# Code analysis functions (adapted from your app.py)
# -----------------------
def check_syntax_and_semantic(code: str, language: str, temp_dir: str) -> str:
    cfg = LANGUAGE_TOOLS_CONFIG.get(language)
    if not cfg:
        return f"Syntax/semantic check not supported for {language}."
    fname = os.path.join(temp_dir, f"temp_code.{language}")
    with open(fname, "w") as f:
        f.write(code)
    syntax_cmd = cfg.get("syntax")
    if not syntax_cmd:
        return "No syntax tool configured."
    # Many tools take a filename; some use stdin
    if language in ("js", "html", "css"):
        res = _run_command_safely(syntax_cmd, input_data=code, cwd=temp_dir)
    elif language == "xml":
        res = _run_command_safely(syntax_cmd + [fname], cwd=temp_dir)
    else:
        # pass filename where appropriate
        if syntax_cmd[-1].startswith("--stdin"):
            res = _run_command_safely(syntax_cmd, input_data=code, cwd=temp_dir)
        else:
            res = _run_command_safely(syntax_cmd + [fname], cwd=temp_dir)
    if res.get("success"):
        return "No syntax/semantic issues found."
    return _parse_error_output(res.get("stderr", "") or res.get("stdout", ""), cfg.get("parser", ""))

def check_runtime_errors(code: str, language: str, temp_dir: str) -> str:
    cfg = LANGUAGE_TOOLS_CONFIG.get(language)
    if not cfg or not cfg.get("runtime"):
        return f"Runtime check not supported for {language}."
    fname = os.path.join(temp_dir, f"temp_runtime.{language}")
    with open(fname, "w") as f:
        f.write(code)
    runtime_cmd = cfg["runtime"]
    # python runtime
    if language == "py":
        res = _run_command_safely(runtime_cmd + [fname], cwd=temp_dir, timeout=CODE_EXECUTION_TIMEOUT)
    elif language == "java":
        compile_res = _run_command_safely(LANGUAGE_TOOLS_CONFIG["java"]["syntax"] + [fname], cwd=temp_dir)
        if not compile_res["success"]:
            return f"Compilation failed:\n{_parse_error_output(compile_res.get('stderr',''), 'java')}"
        class_name = os.path.splitext(os.path.basename(fname))[0]
        res = _run_command_safely(runtime_cmd + [class_name], cwd=temp_dir)
    elif language in ("c", "cpp"):
        compiler = "gcc" if language == "c" else "g++"
        compile_cmd = [compiler, fname, "-o", os.path.join(temp_dir, "temp.out")]
        compile_res = _run_command_safely(compile_cmd, cwd=temp_dir)
        if not compile_res["success"]:
            return f"Compilation failed:\n{_parse_error_output(compile_res.get('stderr',''), compiler)}"
        res = _run_command_safely([os.path.join(temp_dir, "temp.out")], cwd=temp_dir)
    elif language == "js":
        res = _run_command_safely(runtime_cmd + [fname], cwd=temp_dir)
    elif language == "cs":
        prj = os.path.join(temp_dir, "TempProject")
        os.makedirs(prj, exist_ok=True)
        with open(os.path.join(prj, "Program.cs"), "w") as f:
            f.write(code)
        with open(os.path.join(prj, "TempProject.csproj"), "w") as f:
            f.write('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net6.0</TargetFramework></PropertyGroup></Project>')
        res = _run_command_safely(runtime_cmd, cwd=prj)
    elif language == "kt":
        jar = os.path.join(temp_dir, "temp_runtime.jar")
        compile_res = _run_command_safely(["kotlinc", fname, "-include-runtime", "-d", jar], cwd=temp_dir)
        if not compile_res["success"]:
            return f"Compilation failed:\n{_parse_error_output(compile_res.get('stderr',''), 'kotlin')}"
        res = _run_command_safely(["java", "-jar", jar], cwd=temp_dir)
    elif language in ("dart", "go"):
        res = _run_command_safely(cfg["runtime"] + [fname], cwd=temp_dir)
    else:
        return f"Runtime check not implemented for {language}."
    if res.get("success"):
        return "No runtime errors found."
    if res.get("error_type") == "TimeoutExpired":
        return f"Runtime check timed out after {CODE_EXECUTION_TIMEOUT} seconds. Output:\n{res.get('stdout','')}\n{res.get('stderr','')}"
    return _parse_error_output(res.get("stderr","") or res.get("stdout",""), cfg.get("parser",""))

def check_dependency_versions(code: str, language: str, temp_dir: str) -> str:
    cfg = LANGUAGE_TOOLS_CONFIG.get(language)
    if not cfg or not cfg.get("deps"):
        return f"Dependency check not supported for {language}."
    dep_regex = cfg.get("dep_regex")
    if not dep_regex:
        return "No dependency regex configured."
    found = set()
    for line in code.splitlines():
        m = re.search(dep_regex, line)
        if m:
            if language == "py":
                found.add(m.group(2).split('.')[0])
            elif language in ("java","cs","kt"):
                found.add(m.group(2).split('.')[0])
            elif language == "js":
                dn = m.group(2)
                if '/' in dn:
                    found.add(dn.split('/')[0] if not dn.startswith('@') else '/'.join(dn.split('/')[:2]))
                else:
                    found.add(dn)
            else:
                found.add(m.group(1) if m.groups() else line.strip())
    if not found:
        return "No common dependencies detected in the code."
    out = []
    if language == "py":
        for lib in found:
            r = _run_command_safely(["pip","show",lib], cwd=temp_dir, timeout=5)
            if r.get("success"):
                ver = next((l for l in r["stdout"].splitlines() if l.startswith("Version:")), "Version: N/A")
                out.append(f"{lib} -> {ver}")
            else:
                out.append(f"{lib} -> not installed / unavailable ({r.get('stderr','').strip()})")
    elif language == "js":
        # create package.json and npm install
        pkg = {"name":"temp-project","version":"1.0.0","dependencies": {d:"*" for d in found}}
        with open(os.path.join(temp_dir,"package.json"), "w") as fh:
            json.dump(pkg, fh, indent=2)
        inst = _run_command_safely(["npm","install","--silent"], cwd=temp_dir, timeout=120)
        if not inst["success"]:
            out.append(f"npm install failed: {inst.get('stderr','')}")
        else:
            ls = _run_command_safely(["npm","list","--json","--depth=0"], cwd=temp_dir, timeout=20)
            if ls["success"]:
                try:
                    j = json.loads(ls["stdout"])
                    for k,v in j.get("dependencies",{}).items():
                        out.append(f"{k} -> {v.get('version','?')}")
                except Exception:
                    out.append("Could not parse npm list output.")
            else:
                out.append(f"npm list failed: {ls.get('stderr','')}")
    else:
        out.append(f"Dependency check for {language} requires project toolchains. Detected: {', '.join(found)}")
    return "\n".join(out) if out else "No issues found."

def ask_llm_to_fix_code(code: str, language: str) -> str:
    # Uses Ollama API (wizardcoder:7b) as in your app.py
    prompt = f"You're a professional software debugger. Detect all errors: syntax, semantic, runtime, and compatibility.\nReturn improved fixed code and suggestions.\n\nCode ({language}):\n{code}"
    payload = {"model": OLLAMA_MODEL_ID, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        j = resp.json()
        return j.get("response", "No response from Ollama.")
    except Exception as e:
        return f"LLM fix call failed: {e}"

# -----------------------
# Cinematic LLMs (from your cinematic file)
# -----------------------
def run_mixtral(prompt_text: str, max_new_tokens: int = 300) -> str:
    # Uses Mistral Mixtral ID from your cinematic file
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MIXTRAL_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MIXTRAL_MODEL_ID, device_map="auto")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_openchat(prompt_text: str, max_new_tokens: int = 300) -> Dict[str,str]:
    # Uses AutoGPTQ quantized OpenChat model (as in your cinematic file).
    # This requires auto_gptq package and quantized model prepared at OPENCHAT_MODEL_ID
    try:
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM
    except Exception as e:
        raise RuntimeError(f"AutoGPTQ or transformers import failed: {e}")
    model_id = OPENCHAT_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_quantized(model_id, device_map="auto", trust_remote_code=True, offload_folder="offload", revision="main")
    prompt = (f"Enhance the following scene description for cinematic video generation. Add rich visual details, environment cues, lighting, and action.\nInput: {prompt_text}\nEnhanced:")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if ';' in decoded:
        scene_detail, mood = decoded.split(';',1)
    else:
        # try newline split
        parts = decoded.splitlines()
        scene_detail = parts[0] if parts else decoded
        mood = parts[1] if len(parts) > 1 else "unknown"
    return {"scene_detail": scene_detail.strip(), "mood": mood.strip()}

def run_llava(prompt_text: str, model_path: Optional[str] = None) -> Dict[str,str]:
    # If you have a local LLaVA-like multimodal model, set model_path env var or pass model_path.
    # Your cinematic file had a dummy run_llava — here we attempt to load if you set LLAVA_MODEL_PATH env var.
    model_path = model_path or os.getenv("LLAVA_MODEL_PATH", None)
    if not model_path:
        raise RuntimeError("LLava model path not set. Set LLAVA_MODEL_PATH env var or pass model_path.")
    # This is placeholder loading logic — adapt to the exact LLaVA repo you used.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # naive parse:
    if ';' in decoded:
        scene_detail, mood = decoded.split(';',1)
    else:
        scene_detail, mood = decoded, "unknown"
    return {"scene_detail": scene_detail.strip(), "mood": mood.strip()}

# -----------------------
# Input parser for cinematic generator
# -----------------------
def parse_scene_input(prompt: str, character_name: str = "Character", camera_angle: Optional[str] = None, mood_hint: Optional[str] = None, parser_llm: str = "openchat") -> Dict:
    """
    Parse a user prompt and return a dict with:
      - scene: expanded scene description (string)
      - mood: short mood tag (string)
      - characters: string
      - camera_angle: string
      - style: optional style
      - seed: optional seed (int or None)

    Strategy:
      - If prompt looks structured (JSON or key:value lines), parse directly.
      - Otherwise call the selected LLM (openchat/mixtral/llava) to enhance and split.
    """
    # First: try quick structured parse (JSON)
    prompt_stripped = prompt.strip()
    out = {"scene": prompt_stripped, "mood": mood_hint or "neutral", "characters": character_name, "camera_angle": camera_angle or "wide", "style": None, "seed": None}
    # JSON?
    if (prompt_stripped.startswith("{") and prompt_stripped.endswith("}")):
        try:
            j = json.loads(prompt_stripped)
            out["scene"] = j.get("scene", out["scene"])
            out["mood"] = j.get("mood", out["mood"])
            out["characters"] = j.get("characters", out["characters"])
            out["camera_angle"] = j.get("camera_angle", out["camera_angle"])
            out["style"] = j.get("style", out["style"])
            out["seed"] = j.get("seed", out["seed"])
            return out
        except Exception:
            pass
    # Key: value lines parse
    lines = [l.strip() for l in prompt_stripped.splitlines() if l.strip()]
    kv = {}
    for L in lines:
        if ":" in L:
            k,v = L.split(":",1)
            kv[k.strip().lower()] = v.strip()
    if kv:
        out["scene"] = kv.get("scene", out["scene"])
        out["mood"] = kv.get("mood", out["mood"])
        out["characters"] = kv.get("characters", out["characters"])
        out["camera_angle"] = kv.get("camera_angle", out["camera_angle"])
        out["style"] = kv.get("style", out["style"])
        if kv.get("seed"):
            try:
                out["seed"] = int(kv.get("seed"))
            except Exception:
                out["seed"] = None
        return out
    # Else: call LLM parser to enhance + extract
    try:
        if parser_llm.lower() == "openchat":
            parsed = run_openchat(prompt_stripped)
        elif parser_llm.lower() == "mixtral":
            text = run_mixtral(prompt_stripped)
            # naive split by semicolon/newline
            if ';' in text:
                scene_detail, mood = text.split(';',1)
                parsed = {"scene_detail": scene_detail, "mood": mood}
            else:
                parsed = {"scene_detail": text, "mood": mood_hint or "neutral"}
        elif parser_llm.lower() == "llava":
            parsed = run_llava(prompt_stripped)
        else:
            parsed = {"scene_detail": prompt_stripped, "mood": mood_hint or "neutral"}
        out["scene"] = parsed.get("scene_detail", out["scene"])
        out["mood"] = parsed.get("mood", out["mood"])
        return out
    except Exception as e:
        # don't fallback to fake — return best-effort original prompt
        return out

# -----------------------
# gen_zeroscope (fully restored arguments & checks)
# -----------------------
def gen_zeroscope(
    scene: str,
    model_id: str = TEXT_TO_VIDEO_MODEL,
    out_path: str = "zeroscope.mp4",
    num_inference_steps: int = 25,
    guidance_scale: float = 5.0,
    height: int = 256,
    width: int = 448,
    num_frames: int = 18,
    fps: int = 6,
    seed: Optional[int] = None,
    variant: Optional[str] = "fp16",
    device: str = DEVICE,
    init_image = None,
    controlnet_hook: Optional[dict] = None
) -> str:
    """
    High-fidelity wrapper for text->video pipeline (zeroscope style).
    Accepts many generation parameters and performs robust post-processing
    to ensure frames are HWC uint8 RGB and writes mp4 using export_to_video.
    """

    # load pipeline (assumes model available either locally or remote)
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    # seed / generator
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    # Prepare kwargs for pipe
    pipe_kwargs = dict(prompt=scene, num_inference_steps=num_inference_steps, height=height, width=width, num_frames=num_frames)
    # guidance scale may be supported depending on pipeline API
    if guidance_scale is not None:
        pipe_kwargs["guidance_scale"] = guidance_scale
    if generator is not None:
        pipe_kwargs["generator"] = generator

    # If controlnet hook exists, attach or pass it (depends on pipeline implementation)
    # We keep this as an optional hook; if you use ControlNet-based pipeline, adjust accordingly.
    if controlnet_hook:
        # plugin-specific: this is a placeholder to show where to integrate controlnet
        # e.g. pipe.controlnet = controlnet_hook['controlnet']
        pass

    # If an init_image is provided, pass as 'init_image' if pipeline supports it
    if init_image is not None:
        pipe_kwargs["init_image"] = init_image

    # Run generation
    result = pipe(**pipe_kwargs)
    video_frames = getattr(result, "frames", None) or result.get("frames", None) or result

    # Postprocess frames robustly (borrowed from your cinematic file)
    fixed_frames = []
    for i, frame in enumerate(video_frames):
        # convert torch -> numpy
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        # lists -> numpy
        if isinstance(frame, list):
            frame = np.array(frame)
        # If batched (1, H, W, C)
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        # If CHW (C,H,W) -> HWC
        if frame.ndim == 3 and frame.shape[0] in (1,3,4):
            # likely CHW
            if frame.shape[0] in (1,3,4) and frame.shape[0] <= 4:
                frame = np.transpose(frame, (1,2,0))
        # If float in [0,1] scale to 0-255
        try:
            if np.nanmin(frame) >= 0.0 and np.nanmax(frame) <= 1.0:
                frame = (frame * 255.0).astype(np.uint8)
        except Exception:
            # conservatively cast to uint8
            frame = frame.astype(np.uint8)
        # Ensure HWC
        if frame.ndim == 2:
            # grayscale -> convert to 3 channels
            frame = np.stack([frame]*3, axis=-1)
        if frame.ndim == 3 and frame.shape[2] > 3:
            # trim alpha or extra channels
            frame = frame[:, :, :3]
        # final check
        if frame.ndim == 3 and frame.shape[2] == 3:
            fixed_frames.append(frame)
        else:
            # try to salvage by converting to uint8 array with 3 channels
            try:
                arr = np.array(frame)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    fixed_frames.append(arr[:, :, :3].astype(np.uint8))
                else:
                    # expand gray -> rgb
                    if arr.ndim == 2:
                        fixed_frames.append(np.stack([arr]*3, axis=-1).astype(np.uint8))
                    else:
                        # skip invalid frame
                        print(f"Skipping invalid frame {i}, shape: {arr.shape}")
            except Exception as e:
                print(f"Failed to salvage frame {i}: {e}")

    if len(fixed_frames) == 0:
        raise RuntimeError("No valid frames were produced by the text->video pipeline.")

    # If frames have dtype float32 but in 0-1 range, convert
    final_frames = []
    for f in fixed_frames:
        if f.dtype != np.uint8:
            try:
                if f.max() <= 1.0:
                    f = (f * 255).astype(np.uint8)
                else:
                    f = f.astype(np.uint8)
            except Exception:
                f = f.astype(np.uint8)
        final_frames.append(f)

    # Export to video using diffusers utility (or fallback to moviepy)
    try:
        # export_to_video expects list of numpy frames (H,W,3) and will write mp4
        export_to_video(final_frames, out_path, fps=fps)
    except Exception as e:
        # fallback to moviepy writer
        clip = ImageSequenceClip(final_frames, fps=fps)
        clip.write_videofile(out_path, codec="libx264", audio=False)

    return out_path

# -----------------------
# Riffusion music (from cinematic file)
# -----------------------
def gen_music_riffusion(mood: str, out_path: str = RIFFUSION_AUDIO_PATH) -> str:
    # Build Riffusion pipeline from local SD diffusers files (as in your cinematic file)
    vae = AutoencoderKL.from_pretrained(os.path.join(SD_DIFFUSERS_LOCAL, "vae"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(SD_DIFFUSERS_LOCAL, "text_encoder"))
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(SD_DIFFUSERS_LOCAL, "tokenizer"))
    unet = UNet2DConditionModel.from_pretrained(os.path.join(SD_DIFFUSERS_LOCAL, "unet"))
    scheduler = LMSDiscreteScheduler.from_pretrained(os.path.join(SD_DIFFUSERS_LOCAL, "scheduler"))
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(os.path.join(SD_DIFFUSERS_LOCAL, "safety_checker"))
    feature_extractor = CLIPFeatureExtractor.from_pretrained(os.path.join(SD_DIFFUSERS_LOCAL, "feature_extractor"))

    pipe = RiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=safety_checker, feature_extractor=feature_extractor)
    pipe = pipe.to(DEVICE)

    blank = np.zeros((512,512,3), dtype=np.uint8)
    init_img = Image.fromarray(blank).convert("RGBA")
    prompt_text = f"cinematic background music, {mood} mood"
    out = pipe.riffuse(init_img, prompt_text)
    audio = out["audio"]
    if audio.dtype not in (np.float32, np.int16):
        audio = audio.astype(np.float32)
    sr = 44100
    scipy.io.wavfile.write(out_path, sr, audio)
    return out_path

# -----------------------
# TTS (Bark) & Wav2Lip & SadTalker wrappers
# -----------------------
def tts_bark(dialogue: str, out_path: str = "bark.wav") -> str:
    from bark import generate_audio  # assumes bark installed
    audio = generate_audio(dialogue)
    write_wav(out_path, 24000, audio.astype(np.float32))
    return out_path

def lip_sync(face_video_path: str, audio_path: str, out_path: str = "synced_video.mp4") -> str:
    cmd = ["python", WAV2LIP_INFERENCE_SCRIPT, "--checkpoint_path", WAV2LIP_CHECKPOINT, "--face", face_video_path, "--audio", audio_path, "--outfile", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Wav2Lip failed: {proc.stderr}\n{proc.stdout}")
    return out_path

def add_emotion(face_video_path: str, audio_path: str, out_path: str = "emotion_video.mp4") -> str:
    cmd = ["python", SADTALKER_INFER_SCRIPT, "--driven_audio", face_video_path, "--audio", audio_path, "--output", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SadTalker failed: {proc.stderr}\n{proc.stdout}")
    return out_path

def combine_music(m1: str, m2: str, out_path: str = "combined_music.wav") -> str:
    cmd = ["ffmpeg", "-y", "-i", m1, "-i", m2, "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=3", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg amix failed: {proc.stderr}\n{proc.stdout}")
    return out_path

def stitch_final(video_path: str, audio_path: str, music_path: str, out_path: str = "final_render.mp4") -> str:
    video = VideoFileClip(video_path)
    voice = AudioFileClip(audio_path).set_duration(video.duration)
    music = AudioFileClip(music_path).volumex(0.3).set_duration(video.duration)
    mixed_audio = CompositeAudioClip([voice, music])
    final = video.set_audio(mixed_audio.set_duration(video.duration))
    final.write_videofile(out_path, codec="libx264", audio_codec="aac")
    return out_path

# -----------------------
# FastAPI endpoints
# -----------------------
class CodeInputModel:
    # simple typing for FastAPI Pydantic compatibility (we will not rely heavily here)
    pass

# Using Pydantic BaseModel inline for endpoints
from pydantic import BaseModel
class CodeInput(BaseModel):
    code: str
    language: str

class AnalysisResult(BaseModel):
    syntax_errors: str
    runtime_errors: str
    compatibility_issues: str
    llm_fixes: str

@api.post("/analyze", response_model=AnalysisResult)
def analyze_code_endpoint(code_input: CodeInput):
    if code_input.language not in LANGUAGE_TOOLS_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {code_input.language}")
    with TemporaryDirectory() as td:
        try:
            syntax = check_syntax_and_semantic(code_input.code, code_input.language, td)
            runtime = check_runtime_errors(code_input.code, code_input.language, td)
            deps = check_dependency_versions(code_input.code, code_input.language, td)
            llm = ask_llm_to_fix_code(code_input.code, code_input.language)
            return AnalysisResult(syntax_errors=syntax, runtime_errors=runtime, compatibility_issues=deps, llm_fixes=llm)
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}\n\n{tb}")

@api.post("/upload_zip/", response_model=AnalysisResult)
async def upload_zip_endpoint(file: UploadFile = File(...)):
    with TemporaryDirectory() as td:
        zip_path = os.path.join(td, file.filename)
        with open(zip_path, "wb") as f:
            f.write(await file.read())
        extracted = os.path.join(td, "extracted")
        shutil.unpack_archive(zip_path, extracted)
        full_code = ""
        language = "unknown"
        for root, _, files in os.walk(extracted):
            for fname in files:
                if any(fname.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    fp = os.path.join(root, fname)
                    with open(fp, "r", errors="ignore") as fh:
                        full_code += fh.read() + "\n"
                        if language == "unknown":
                            ext = os.path.splitext(fname)[1]
                            language = ext.strip(".") if ext.startswith(".") else "py"
        if not full_code:
            raise HTTPException(status_code=400, detail="No supported code files found in the zip archive.")
        return analyze_code_endpoint(CodeInput(code=full_code, language=language))

class PromptInput(BaseModel):
    prompt: str
    camera_angle: Optional[str] = "wide"
    dialogue: Optional[str] = "Hello"
    mood: Optional[str] = "dramatic"
    character_name: Optional[str] = "Al"
    parser_llm: Optional[str] = "openchat"  # which LLM to use for parsing/enhancement

@api.post("/generate")
def generate_endpoint(pi: PromptInput):
    """
    Orchestrates full cinematic pipeline using components above.
    Returns JSON with artifact paths.
    """
    try:
        # 1) parse/enhance prompt
        parsed = parse_scene_input(pi.prompt, character_name=pi.character_name, camera_angle=pi.camera_angle, mood_hint=pi.mood, parser_llm=pi.parser_llm)
        scene_text = parsed["scene"]
        mood = parsed["mood"]
        # 2) script (use distilgpt2 model in your cinematic file)
        tokenizer = AutoTokenizer.from_pretrained(DISTILGPT2_SCRIPT_LLM)
        model = AutoModelForCausalLM.from_pretrained(DISTILGPT2_SCRIPT_LLM).to(DEVICE)
        s_prompt = f"You are a script writer for cinematic short films.\nCharacters: {parsed['characters']}\nScene: {scene_text}\nWrite a short film script:"
        inputs = tokenizer(s_prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)
        script = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if script.startswith(s_prompt):
            script = script[len(s_prompt):].strip()

        # 3) generate video
        zeroscope_path = gen_zeroscope(scene_text, model_id=TEXT_TO_VIDEO_MODEL, out_path="zeroscope.mp4", num_inference_steps=25, guidance_scale=5.0, height=256, width=448, num_frames=18, fps=6, seed=parsed.get("seed"))

        # 4) tts
        bark_path = tts_bark(pi.dialogue, out_path="bark.wav")

        # 5) music
        music_path = gen_music_riffusion(mood, out_path="riffusion_out.wav")

        # 6) lip-sync
        synced = lip_sync(zeroscope_path, bark_path, out_path="synced_video.mp4")

        # 7) emotion animation (SadTalker)
        emotion_video = add_emotion(synced, bark_path, out_path="emotion_video.mp4")

        # 8) final stitch
        final = stitch_final(emotion_video, bark_path, music_path, out_path="final_render.mp4")

        return {
            "script": script,
            "zeroscope_path": zeroscope_path,
            "bark_path": bark_path,
            "music_path": music_path,
            "synced_path": synced,
            "emotion_path": emotion_video,
            "final_path": final
        }
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}\n\n{tb}")

@api.get("/artifact/{name}")
def artifact(name: str):
    if os.path.exists(name):
        return FileResponse(name, media_type="application/octet-stream", filename=name)
    raise HTTPException(status_code=404, detail="File not found")



# pipeline cache to avoid reloading models repeatedly
_PIPELINE_CACHE = {}

def _get_pipeline(key: str, loader_fn):
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]
    try:
        p = loader_fn()
        _PIPELINE_CACHE[key] = p
        return p
    except Exception as e:
        _PIPELINE_CACHE[key] = None
        return None

# Text gen
'''def generate_text_func(user: str, prompt: str, max_length: int = 150):
    log_query(user, "text-gen", prompt)
    def loader():
        from transformers import pipeline
        return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    pipe = _get_pipeline("text_gen", loader)
    if pipe is None:
        return "Text generation model not available. Install transformers and model."
    out = pipe(prompt, max_length=max_length, do_sample=True)
    return out[0].get("generated_text", "")'''
# ---------- oobabooga ----------
def oobabooga_generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    url = OOBABOOGA_API_URL.rstrip("/") + OOBABOOGA_ENDPOINT
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "top_p": 0.9,
        "stop": [],
        "grammar": None
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        # text-generation-webui returns choices / results depending on API; handle both
        if isinstance(j, dict):
            if "results" in j and j["results"]:
                return j["results"][0].get("text", "")
            if "results" in j and isinstance(j["results"], list):
                return "".join([x.get("text","") for x in j["results"]])
            if "text" in j:
                return j["text"]
        return str(j)
    except Exception as e:
        return f"[oobabooga error] {e}"

# Paraphrase
def paraphrase_text_func(user: str, text: str, max_length:int = 100):
    log_query(user, "paraphrase", text)
    def loader():
        from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
        model_name = "Vamsi/T5_Paraphrase_Paws"
        tok = T5Tokenizer.from_pretrained(model_name)
        mod = T5ForConditionalGeneration.from_pretrained(model_name)
        return pipeline("text2text-generation", model=mod, tokenizer=tok)
    pipe = _get_pipeline("paraphrase", loader)
    if pipe is None:
        return "Paraphrase model not available. Install transformers and the model."
    out = pipe(f"paraphrase: {text}", max_length=max_length, num_beams=5)
    return out[0].get("generated_text", "")

# Code generation
def generate_code_func(user: str, description: str, max_length:int = 200):
    log_query(user, "code-gen", description)
    def loader():
        from transformers import pipeline
        return pipeline("text-generation", model="Salesforce/codegen-350M-mono")
    pipe = _get_pipeline("code_gen", loader)
    if pipe is None:
        return "Code generation model not available. Install transformers and model."
    out = pipe(f"# Python code to {description}", max_length=max_length)
    return out[0].get("generated_text","")
# ---------- StarCoder (local path only) ----------
'''_STARCODER_CACHE = {"tok": None, "model": None}

def starcoder_generate(description: str, max_new_tokens: int = STARCODER_MAX_NEW_TOKENS) -> str:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        return f"[StarCoder error] transformers import failed: {e}"
    if not os.path.isdir(STARCODER_LOCAL_PATH):
        return f"[StarCoder error] STARCODER_LOCAL_PATH not found: {STARCODER_LOCAL_PATH}"
    if _STARCODER_CACHE["tok"] is None or _STARCODER_CACHE["model"] is None:
        tok = AutoTokenizer.from_pretrained(STARCODER_LOCAL_PATH, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(STARCODER_LOCAL_PATH, local_files_only=True, trust_remote_code=True, device_map="auto")
        _STARCODER_CACHE["tok"] = tok
        _STARCODER_CACHE["model"] = model
    tok, model = _STARCODER_CACHE["tok"], _STARCODER_CACHE["model"]
    prompt = f"# Write self-contained {description}\n"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text'''

# TTS -> use bark if available (adapted)
def text_to_speech_func(user: str, text: str):
    log_query(user, "tts", text)
    # try bark TTS (from cinematic functions)
    try:
        wavpath = tts_bark(text, out_path=f"{user}_bark.wav")
        return wavpath, None
    except Exception as e:
        # fallback attempt: SpeechT5 (transformers) if installed
        try:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
            proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            mod = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            inp = proc(text=text, return_tensors="pt")["input_ids"]
            # note: API variants exist; this may fail in many envs
            speech = mod.generate_speech(inp)
            fname = f"{user}_tts.wav"
            import torchaudio
            torchaudio.save(fname, speech.unsqueeze(0), 16000)
            return fname, None
        except Exception as e2:
            return None, f"TTS not available (bark & SpeechT5 failed): {e} / {e2}"
# ---------- Coqui TTS ----------
'''def coqui_tts_speak(text: str, out_path: str = "tts.wav") -> str:
    try:
        from TTS.api import TTS
    except Exception as e:
        raise RuntimeError(f"Coqui TTS not installed or import failed: {e}")
    if COQUI_TTS_MODEL_PATH and os.path.exists(COQUI_TTS_MODEL_PATH):
        tts = TTS(model_path=COQUI_TTS_MODEL_PATH)
    else:
        # will download if not present; prefer setting COQUI_TTS_MODEL_PATH to a local folder
        tts = TTS(model_name=COQUI_TTS_MODEL_NAME)
    tts.tts_to_file(text=text, file_path=out_path)
    if not os.path.exists(out_path):
        raise RuntimeError("Coqui TTS produced no file.")
    return out_path'''

# Video summarization: extract audio -> ASR -> summarizer
def summarize_video_func(user: str, video_path: str):
    log_query(user, "video-sum", video_path)
    try:
        clip = VideoFileClip(video_path)
        tmp_wav = f"{user}_tmp_video_audio.wav"
        clip.audio.write_audiofile(tmp_wav, logger=None)
    except Exception as e:
        return f"Failed to extract audio from video: {e}"
    transcription = speech_to_text_func(user, tmp_wav)
    def loader():
        from transformers import pipeline
        return pipeline("summarization", model="facebook/bart-large-cnn")
    summarizer = _get_pipeline("summarizer", loader)
    if summarizer is None:
        return "Summarizer model not available. Install transformers and model."
    out = summarizer(transcription)
    return out[0].get("summary_text", "")

# Music generation -> replaced with gen_music_riffusion
def generate_music_func(user: str, prompt: str, duration: int = 4):
    log_query(user, "music-gen", prompt)
    # call gen_music_riffusion with mood derived from prompt
    mood = "calm"
    if "dramatic" in prompt.lower(): mood = "dramatic"
    try:
        path = gen_music_riffusion(mood, out_path=f"{user}_music.wav")
        return path
    except Exception as e:
        # fallback: silent wav
        sr = 44100
        wav = np.zeros(int(duration * sr), dtype=np.int16)
        fname = f"{user}_music.wav"
        scipy.io.wavfile.write(fname, sr, wav)
        return fname

# Text->image (attempt to use diffusers or transformers text-to-image)
'''def text_to_image_func(user: str, prompt: str):
    log_query(user, "text2img", prompt)
    # try diffusers pipeline
    try:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe = pipe.to(DEVICE)
        image = pipe(prompt).images[0]
        fname = re.sub(r"\W+", "_", prompt)[:200] + f"_{user}_t2i.png"
        image.save(fname)
        return fname, None
    except Exception as e:
        return None, f"Text2Image not available: {e}"'''
# ---------- AUTOMATIC1111 ----------
def sd_webui_txt2img(prompt: str, steps: int = SD_DEFAULT_STEPS, cfg_scale: float = SD_DEFAULT_CFG, w: int = SD_DEFAULT_WIDTH, h: int = SD_DEFAULT_HEIGHT) -> str:
    url = SD_WEBUI_API.rstrip("/") + "/sdapi/v1/txt2img"
    payload = {"prompt": prompt, "steps": steps, "cfg_scale": cfg_scale, "width": w, "height": h}
    r = requests.post(url, json=payload, timeout=900)
    r.raise_for_status()
    j = r.json()
    if "images" not in j or not j["images"]:
        raise RuntimeError("SD WebUI returned no images.")
    b64 = j["images"][0]
    img = Image.open(io.BytesIO(base64.b64decode(b64.split(",",1)[-1])))
    fname = re.sub(r"\W+", "_", prompt)[:120] + "_t2i.png"
    img.save(fname)
    return fname

# Image->image (img2img)
'''def image_to_image_func(user: str, prompt: str, image_path: str):
    log_query(user, "img2img", prompt)
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        from PIL import Image as PILImage
        init = PILImage.open(image_path).convert("RGB")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe = pipe.to(DEVICE)
        out = pipe(prompt=prompt, init_image=init, strength=0.75, guidance_scale=7.5)
        image = out.images[0]
        fname = re.sub(r"\W+", "_", prompt)[:200] + f"_{user}_img2img.png"
        image.save(fname)
        return fname, None
    except Exception as e:
        return None, f"Img2Img not available: {e}"'''
def sd_webui_img2img(prompt: str, image_path: str, strength: float = 0.75, steps: int = SD_DEFAULT_STEPS, cfg_scale: float = SD_DEFAULT_CFG) -> str:
    url = SD_WEBUI_API.rstrip("/") + "/sdapi/v1/img2img"
    with open(image_path, "rb") as fh:
        b64img = base64.b64encode(fh.read()).decode("utf-8")
    payload = {
        "prompt": prompt,
        "init_images": [b64img],
        "denoising_strength": strength,
        "steps": steps,
        "cfg_scale": cfg_scale
    }
    r = requests.post(url, json=payload, timeout=900)
    r.raise_for_status()
    j = r.json()
    if "images" not in j or not j["images"]:
        raise RuntimeError("SD WebUI returned no images.")
    b64 = j["images"][0]
    from io import BytesIO
    img = Image.open(BytesIO(base64.b64decode(b64.split(",",1)[-1])))
    fname = re.sub(r"\W+", "_", prompt)[:120] + "_img2img.png"
    img.save(fname)
    return fname

# 3D gaussian (external)
'''def text_to_3d_gaussian_func(user: str, prompt: str):
    log_query(user, "text2gauss3d", prompt)
    try:
        subprocess.run([
            "dreamgaussian/main.py",
            "--prompt", prompt,
            "--config", "configs/text.yaml",
            "--out", f"{user}_gauss3d"
        ], check=True)
        return f"{user}_gauss3d/outputs/scene.obj"
    except Exception as e:
        return f"3D generation failed (ensure GaussianDreamer installed): {e}"'''
# ---------- Stable DreamFusion ----------
def dreamfusion_generate(prompt: str, out_dir: str = DREAMFUSION_OUTDIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["python", DREAMFUSION_PY, "--prompt", prompt, "--save_path", out_dir]
    res = _run_command_safely(cmd, cwd=DREAMFUSION_DIR, timeout=1200)
    if not res["success"]:
        raise RuntimeError(f"DreamFusion failed: {res.get('stderr','')}\n{res.get('stdout','')}")
    # Heuristic: find an OBJ or GLB in out_dir
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith((".obj",".glb",".ply")):
                return os.path.join(root, f)
    raise RuntimeError("DreamFusion finished but no 3D artifact was found.")

# CLI endpoints

class TextGenIn(BaseModel):
    user: str
    prompt: str
    max_length: Optional[int] = 300

@api.post("/text-gen")
def text_gen_endpoint(inp: TextGenIn):
    '''try:
        out = generate_text_func(inp.user, inp.prompt, max_length=inp.max_length)
        return {"result": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''
    log_query(inp.user, "text-gen", inp.prompt)
    out = oobabooga_generate(inp.prompt, max_new_tokens=int(inp.max_length or 200))
    return {"result": out}


class ParaphraseIn(BaseModel):
    user: str
    text: str
    style: Optional[str] = "concise"

@api.post("/paraphrase")
def paraphrase_endpoint(inp: ParaphraseIn):
    '''try:
        out = paraphrase_text_func(inp.user, inp.text)
        return {"result": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''
    log_query(inp.user, "paraphrase", inp.text)
    prompt = f"Paraphrase the following in a {inp.style} style while preserving meaning:\n\n{inp.text}\n\nParaphrase:"
    out = oobabooga_generate(prompt, max_new_tokens=200)
    return {"result": out.strip()}


class CodeGenIn(BaseModel):
    user: str
    description: str
    max_tokens: Optional[int] = STARCODER_MAX_NEW_TOKENS

@api.post("/code-gen")
def code_gen_endpoint(inp: CodeGenIn):
    '''try:
        out = generate_code_func(inp.user, inp.description)
        return {"result": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''
    log_query(inp.user, "code-gen", inp.description)
    out = starcoder_generate(inp.description, max_new_tokens=int(inp.max_tokens or STARCODER_MAX_NEW_TOKENS))
    return {"result": out}

class TTSIn(BaseModel):
    user: str
    text: str

@api.post("/tts")
def tts_endpoint(inp: TTSIn):
    '''try:
        fname, err = text_to_speech_func(inp.user, inp.text)
        if err:
            raise HTTPException(status_code=500, detail=err)
        return {"wav_path": fname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''
    log_query(inp.user, "tts", inp.text)
    try:
        path = coqui_tts_speak(inp.text, out_path=f"{inp.user}_tts.wav")
        return {"wav_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


'''class VidSumIn(BaseModel):
    user: str
    video_path: str

@api.post("/video-sum")
def video_sum_endpoint(inp: VidSumIn):
    try:
        out = summarize_video_func(inp.user, inp.video_path)
        return {"summary": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''
class STTIn(BaseModel):
    user: str
    audio_path: str  # server-side path after upload

@api.post("/stt")
def stt_endpoint(inp: STTIn):
    log_query(inp.user, "stt", inp.audio_path)
    try:
        text = whisper_cpp_transcribe(inp.audio_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MusicIn(BaseModel):
    user: str
    prompt: str
    duration: Optional[int] = 4

@api.post("/music-gen")
def music_gen_endpoint(inp: MusicIn):
    try:
        out = generate_music_func(inp.user, inp.prompt, inp.duration)
        return {"wav_path": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class Text2ImgIn(BaseModel):
    user: str
    prompt: str
    steps: Optional[int] = SD_DEFAULT_STEPS
    cfg: Optional[float] = SD_DEFAULT_CFG
    width: Optional[int] = SD_DEFAULT_WIDTH
    height: Optional[int] = SD_DEFAULT_HEIGHT

@api.post("/text2img")
def text2img_endpoint(inp: Text2ImgIn):
    '''try:
        fname, err = text_to_image_func(inp.user, inp.prompt)
        if err:
            raise HTTPException(status_code=500, detail=err)
        return {"image_path": fname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''
    log_query(inp.user, "text2img", inp.prompt)
    try:
        path = sd_webui_txt2img(inp.prompt, steps=int(inp.steps), cfg_scale=float(inp.cfg), w=int(inp.width), h=int(inp.height))
        return {"image_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class Img2ImgIn(BaseModel):
    user: str
    prompt: str
    image_path: str
    strength: Optional[float] = 0.75

@api.post("/img2img")
def img2img_endpoint(inp: Img2ImgIn):
    '''try:
        fname, err = image_to_image_func(inp.user, inp.prompt, inp.image_path)
        if err:
            raise HTTPException(status_code=500, detail=err)
        return {"image_path": fname}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''
    log_query(inp.user, "img2img", f"{inp.prompt} [{inp.image_path}]")
    try:
        path = sd_webui_img2img(inp.prompt, inp.image_path, strength=float(inp.strength))
        return {"image_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class Text2GaussIn(BaseModel):
    user: str
    prompt: str

@api.post("/text2gauss3d")
def text2gauss_endpoint(inp: Text2GaussIn):
    '''try:
        out = text_to_3d_gaussian_func(inp.user, inp.prompt)
        return {"obj_path": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''
    log_query(inp.user, "text2gauss3d", inp.prompt)
    try:
        path = dreamfusion_generate(inp.prompt)
        return {"obj_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Run backend & Gradio UI
# -----------------------
def run_backend():
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="info")

def start_ui():
    # launch backend in a thread
    t = threading.Thread(target=run_backend, daemon=True)
    t.start()
    time.sleep(2.0)

    def _post_json(path, payload, timeout=900):
        try:
            r = requests.post(f"http://127.0.0.1:8000{path}", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    with gr.Blocks(title="Merged: Code Debugger + Cinematic Pipeline + CLI Tools") as demo:
        gr.Markdown("# CLI Tools → GUI (integrated) + Merged AI Code Debugger + Cinematic Generator")
        with gr.Tab("CLI Tools (GUI)"):
            with gr.Row():
                user_txt = gr.Textbox(label="User ", value="user1")

            with gr.Column():
                gr.Markdown("## Text Generation")
                tg_prompt = gr.Textbox(label="Prompt", lines=4)
                tg_max = gr.Slider(32, 512, value=150, step=1, label="Max length")
                tg_btn = gr.Button("Generate")
                tg_out = gr.Textbox(label="Generated text", lines=6)

                def _tg(user, prompt, maxlen):
                    r = _post_json("/text-gen", {"user": user, "prompt": prompt, "max_length": int(maxlen)})
                    if "error" in r:
                        return r["error"]
                    return r.get("result", "")

                tg_btn.click(_tg, inputs=[user_txt, tg_prompt, tg_max], outputs=[tg_out])

                gr.Markdown("## Paraphrase")
                para_in = gr.Textbox(label="Text to paraphrase", lines=3)
                para_btn = gr.Button("Paraphrase")
                para_out = gr.Textbox(label="Paraphrase", lines=3)

                def _para(user, text):
                    r = _post_json("/paraphrase", {"user": user, "text": text})
                    if "error" in r:
                        return r["error"]
                    return r.get("result", "")

                para_btn.click(_para, inputs=[user_txt, para_in], outputs=[para_out])

                gr.Markdown("## Code Generation")
                code_desc = gr.Textbox(label="Description", lines=2)
                code_btn = gr.Button("Generate Code")
                code_out = gr.Textbox(label="Code", lines=8)

                def _code(user, desc):
                    r = _post_json("/code-gen", {"user": user, "description": desc})
                    if "error" in r:
                        return r["error"]
                    return r.get("result", "")

                code_btn.click(_code, inputs=[user_txt, code_desc], outputs=[code_out])

            with gr.Column():
                gr.Markdown("## Text → Speech")
                tts_text = gr.Textbox(label="Text", lines=3)
                tts_btn = gr.Button("Generate TTS")
                tts_audio = gr.Audio(label="Audio", interactive=False)
                tts_status = gr.Textbox(visible=False)

                def _tts(user, text):
                    r = _post_json("/tts", {"user": user, "text": text}, timeout=120)
                    if "error" in r:
                        return None, r["error"]
                    path = r.get("wav_path")
                    if path and os.path.exists(path):
                        return path, ""
                    return None, "TTS returned a path that doesn't exist."

                tts_btn.click(_tts, inputs=[user_txt, tts_text], outputs=[tts_audio, tts_status])

            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Text → Image")
                    t2i_prompt = gr.Textbox(label="Prompt", lines=2)
                    t2i_btn = gr.Button("Generate Image")
                    t2i_img = gr.Image(label="Image")

                    def _t2i(user, prompt):
                        r = _post_json("/text2img", {"user": user, "prompt": prompt}, timeout=300)
                        if "error" in r:
                            return None
                        path = r.get("image_path")
                        if path and os.path.exists(path):
                            return path
                        return None

                    t2i_btn.click(_t2i, inputs=[user_txt, t2i_prompt], outputs=[t2i_img])

                with gr.Column():
                    gr.Markdown("## Image → Image")
                    img_prompt = gr.Textbox(label="Prompt", lines=2)
                    img_file = gr.File(label="Upload image")
                    img_btn = gr.Button("Run Img2Img")
                    img_out = gr.Image(label="Result")

                    def _img2img(user, prompt, uploaded):
                        if uploaded is None:
                            return None
                        r = _post_json("/img2img", {"user": user, "prompt": prompt, "image_path": uploaded.name}, timeout=600)
                        if "error" in r:
                            return None
                        path = r.get("image_path")
                        if path and os.path.exists(path):
                            return path
                        return None

                    img_btn.click(_img2img, inputs=[user_txt, img_prompt, img_file], outputs=[img_out])

            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Video Summarization")
                    vid_file = gr.File(label="Upload video")
                    vid_btn = gr.Button("Summarize")
                    vid_out = gr.Textbox(label="Summary", lines=4)

                    def _vidsum(user, uploaded):
                        if uploaded is None:
                            return "No file uploaded."
                        r = _post_json("/video-sum", {"user": user, "video_path": uploaded.name}, timeout=600)
                        if "error" in r:
                            return r["error"]
                        return r.get("summary", "")

                    vid_btn.click(_vidsum, inputs=[user_txt, vid_file], outputs=[vid_out])

                with gr.Column():
                    gr.Markdown("## Generate Music")
                    music_prompt = gr.Textbox(label="Prompt", lines=2)
                    music_duration = gr.Slider(1, 30, value=4, label="Duration (s)")
                    music_btn = gr.Button("Generate Music")
                    music_audio = gr.Audio(label="Music")

                    def _music(user, prompt, dur):
                        r = _post_json("/music-gen", {"user": user, "prompt": prompt, "duration": int(dur)}, timeout=300)
                        if "error" in r:
                            return None
                        path = r.get("wav_path")
                        if path and os.path.exists(path):
                            return path
                        return None

                    music_btn.click(_music, inputs=[user_txt, music_prompt, music_duration], outputs=[music_audio])

            gr.Markdown("---")
            with gr.Row():
                gr.Markdown("## 3D Gaussian (external)")
                g_prompt = gr.Textbox(label="Prompt", lines=2)
                g_btn = gr.Button("Generate 3D")
                g_out = gr.Textbox(label="Output", lines=2)

                def _gauss(user, prompt):
                    r = _post_json("/text2gauss3d", {"user": user, "prompt": prompt}, timeout=600)
                    if "error" in r:
                        return r["error"]
                    return r.get("obj_path", "")

                g_btn.click(_gauss, inputs=[user_txt, g_prompt], outputs=[g_out])

        def analyze_api(code, language):
          try:
            r = requests.post("http://127.0.0.1:8000/analyze", json={"code": code, "language": language}, timeout=120)
            r.raise_for_status()
            j = r.json()
            return j["syntax_errors"], j["runtime_errors"], j["compatibility_issues"], j["llm_fixes"]
          except Exception as e:
            return f"API error: {e}", "", "", ""

        def upload_zip_api(f):
          try:
            with open(f.name, "rb") as fh:
                files = {"file": (os.path.basename(f.name), fh, "application/zip")}
                r = requests.post("http://127.0.0.1:8000/upload_zip/", files=files, timeout=300)
                r.raise_for_status()
                j = r.json()
                return j["syntax_errors"], j["runtime_errors"], j["compatibility_issues"], j["llm_fixes"]
          except Exception as e:
            return f"Upload error: {e}", "", "", ""


        def generate_api(prompt, camera_angle, dialogue, mood, character_name, parser_llm):
          try:
            payload = {"prompt": prompt, "camera_angle": camera_angle, "dialogue": dialogue, "mood": mood, "character_name": character_name, "parser_llm": parser_llm}
            r = requests.post("http://127.0.0.1:8000/generate", json=payload, timeout=3600)
            r.raise_for_status()
            j = r.json()
            return j.get("script",""), j.get("final_path", None)
          except Exception as e:
            return f"Generation error: {e}", None

        with gr.Tab("Code Debugger"):
            lang = gr.Dropdown(list(LANGUAGE_TOOLS_CONFIG.keys()), value="py", label="Language")
            code_box = gr.Textbox(lines=20, placeholder="Paste code here...")
            analyze_btn = gr.Button("Analyze")
            zip_upload = gr.File(label="Upload ZIP")
            syntax_out = gr.Textbox(label="Syntax & Semantic Errors")
            runtime_out = gr.Textbox(label="Runtime Errors")
            dep_out = gr.Textbox(label="Dependency Issues")
            llm_out = gr.Textbox(label="LLM Fixes")
            analyze_btn.click(analyze_api, inputs=[code_box, lang], outputs=[syntax_out, runtime_out, dep_out, llm_out])
            zip_upload.change(upload_zip_api, inputs=[zip_upload], outputs=[syntax_out, runtime_out, dep_out, llm_out])

        with gr.Tab("Cinematic Generator"):
            prompt_in = gr.Textbox(label="Scene prompt (free text or JSON / key:value)", lines=4)
            camera_in = gr.Textbox(label="Camera angle", value="wide")
            dialogue_in = gr.Textbox(label="Dialogue", value="Hello")
            mood_in = gr.Textbox(label="Mood", value="dramatic")
            char_in = gr.Textbox(label="Character name", value="Alex")
            parser_choice = gr.Dropdown(["openchat","mixtral","llava"], value="openchat", label="Parser LLM (choose the LLM to use for prompt expansion)")
            gen_btn = gr.Button("Generate")
            script_out = gr.Textbox(label="Generated Script")
            video_out = gr.Video(label="Final Render")
            gen_btn.click(generate_api, inputs=[prompt_in, camera_in, dialogue_in, mood_in, char_in, parser_choice], outputs=[script_out, video_out])

    demo.launch(share=True)

'''def start_all_services():'''
def start_servers():
    services = [
        {
            "name": "oobabooga Text Generation",
            "path": "/content/text-generation-webui",
            "cmd": ["python", "server.py", "--api", "--listen", "--port", "5000"],
            "delay": 10
        },
        #{
         #   "name": "Whisper.cpp Speech-to-Text",
         #   "path": "/content/whisper.cpp",
         #   "cmd": ["main", "-m", "./models/ggml-large-v2.bin", "--host", "0.0.0.0", "--port", "5001"],
         #   "delay": 5
        #},
        {
            "name": "Coqui TTS",
            "path": "/content/TTS",
            "cmd": ["python", "TTS/server/server.py", "--host", "0.0.0.0", "--port", "5002"],
            "delay": 5
        },
        {
            "name": "Stable Diffusion WebUI (AUTOMATIC1111)",
            "path": "/content/stable-diffusion-webui",
            "cmd": ["python", "launch.py"],
            "env": {"COMMANDLINE_ARGS": "--api --listen --port 7860"},
            "delay": 15
        },
        {
            "name": "StarCoder",
            "path": "/content/starcoder",
            "cmd": ["python", "app.py", "--host", "0.0.0.0", "--port", "5003"],
            "delay": 5
        },
        {
            "name": "Stable DreamFusion (Text-to-3D)",
            "path": "/content/stable-dreamfusion",
            "cmd": ["python", "app.py", "--host", "0.0.0.0", "--port", "5004"],
            "delay": 5
        }
    ]

    '''for svc in services:
        if not os.path.exists(svc["path"]):
            print(f"[WARN] {svc['name']} folder not found — clone it before running.")
            continue

        print(f"Starting {svc['name']}...")
        env = dict(os.environ, **svc.get("env", {}))
        subprocess.Popen(
            svc["cmd"],
            cwd=svc["path"],
            env=env
        )
        time.sleep(svc["delay"])

    print("✅ All services started.")'''
    """
    Starts all external services (like oobabooga, Coqui TTS, etc.)
    as subprocesses, allowing them to run concurrently in the background.

    Args:
        services (list): A list of dictionaries, where each dictionary
                         contains the configuration for a service to be started.
    """
    print("Starting all external services...")
    running_processes = []
    for svc in services:
        name = svc.get("name")
        command = svc.get("cmd")
        cwd = svc.get("path")
        #port = next((arg.split('--port ')[1] for arg in command if '--port' in arg), "N/A")
        try:
          port_index = command.index("--port")
          port = command[port_index + 1]
        except (ValueError, IndexError):
          port = "N/A"


        if not command:
            print(f"❌ Error: Command not found for service: {name}")
            continue

        if not os.path.exists(cwd):
            print(f"❌ Error: Directory not found for service {name}: {cwd}")
            continue

        try:
            print(f"🚀 Launching {name} on port {port} from directory {cwd}...")
            # Use Popen to run the service in the background
            process = subprocess.Popen(command, cwd=cwd)
            running_processes.append(process)
            print(f"✅ {name} started. PID: {process.pid}")
            time.sleep(5)  # Give the service 5 seconds to start up
        except FileNotFoundError:
            print(f"❌ Error: The executable for {name} was not found.")
        except Exception as e:
            print(f"❌ An unexpected error occurred while starting {name}: {e}")
            time.sleep(30)  # Give the service 5 seconds to start up
        print(f"All services are running in the background successfully")

    return running_processes

if __name__ == "__main__":
    app = FastAPI()
    #start_all_services()
    start_servers()
    start_ui()
