import os
import sys
from pathlib import Path
from ui import build_ui
import requests
from packaging import version as version_parser
import gradio as gr
from utils import load_settings  # <--- Added Import

SCRIPT_VERSION = "5.0.0" 
VERSION_URL = "https://raw.githubusercontent.com/fureys52-oss/Anki-Creator-v2.0.0/refs/heads/main/version.txt" 
DOWNLOAD_URL = "https://github.com/fureys52-oss/Anki-Creator-v2.0.0/archive/refs/heads/main/zip"

# --- Global Configuration ---
PDF_CACHE_DIR = Path(".pdf_cache")
AI_CACHE_DIR = Path(".ai_cache")
LOG_DIR = Path("logs")
MAX_LOG_FILES = 10
MAX_DECKS = 100

def check_for_updates():
    return gr.update(visible=False)

def load_clip_model():
    """Loads Local AI models explicitly on CPU."""
    models = {'clip': None, 'text': None}
    try:
        from sentence_transformers import SentenceTransformer
        
        models['clip'] = SentenceTransformer('clip-ViT-B-32', device='cpu')
        # [CHANGE] Enforce MPNet here
        models['text'] = SentenceTransformer('all-mpnet-base-v2', device='cpu') 
        return {'model': models}, "✅ Local AI Ready"
    except Exception as e:
        return {'model': None}, f"❌ Model Load Error: {e}"

if __name__ == "__main__":
    # 1. Load Settings from disk
    current_settings = load_settings()

    # 2. Define Cache Dirs
    cache_dirs = (PDF_CACHE_DIR, AI_CACHE_DIR)
    
    # 3. Launch UI with Settings passed explicitly
    app = build_ui(
        settings=current_settings,          # <--- FIXED: Passed Settings
        version=SCRIPT_VERSION,
        max_decks=MAX_DECKS,
        cache_dirs=cache_dirs,
        log_dir=LOG_DIR,
        max_log_files=MAX_LOG_FILES,
        update_checker_func=check_for_updates,
        load_clip_model_func=load_clip_model
    )
    
    app.launch(server_name="127.0.0.1", inbrowser=True, favicon_path="icon.ico")