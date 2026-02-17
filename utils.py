import os
import shutil
import re
from pathlib import Path
from typing import List, Any
import gradio as gr
import pytesseract
import sys
import json
import platform
import psutil
import shutil
import subprocess
from PIL import Image
import io
import networkx as nx
import numpy as np
from sentence_transformers import util
import os
from datetime import datetime
import threading
import requests

import os
import shutil
import re
from pathlib import Path
import json
import platform
import subprocess
import threading
import sys
import io
from datetime import datetime

# --- THIRD PARTY IMPORTS ---
# These are critical. If the app crashes, it's usually because one of these is missing.
try:
    import psutil
    import requests
    import gradio as gr
    from PIL import Image
    import pytesseract
    import networkx as nx
    import numpy as np
    import torch
    from sentence_transformers import util, CrossEncoder
except ImportError as e:
    print(f"CRITICAL ERROR: Missing library {e}. Run 'pip install -r requirements.txt'")

def configure_tesseract():
    """Checks for Tesseract binary."""
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
        if platform.system() == "Windows":
            executable_name = "tesseract.exe"
            platform_dir = "windows"
        elif platform.system() == "Darwin":
            executable_name = "tesseract"
            platform_dir = "macos"
        else:
            executable_name = "tesseract"
            platform_dir = "linux"

        tesseract_path = os.path.join(application_path, 'binaries', platform_dir, executable_name)
        tessdata_dir = os.path.join(application_path, 'binaries', platform_dir, 'tessdata')

        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            return True
    
    if shutil.which("tesseract"):
        return True
    if sys.platform == "win32":
        windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(windows_path):
            pytesseract.pytesseract.tesseract_cmd = windows_path
            return True

    return False

def get_system_specs():
    specs = {"ram_gb": 0, "gpu_vram_gb": 0, "has_nvidia": False, "is_mac": False}
    try:
        specs["ram_gb"] = round(psutil.virtual_memory().total / (1024**3))
        if platform.system() == "Darwin" and platform.processor() == "arm":
            specs["is_mac"] = True
            specs["gpu_vram_gb"] = specs["ram_gb"] 
        elif shutil.which("nvidia-smi"):
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], 
                    encoding="utf-8"
                )
                specs["gpu_vram_gb"] = sum([int(x) for x in output.strip().split('\n')]) / 1024
                specs["has_nvidia"] = True
            except: pass
    except: pass
    return specs

def recommend_local_model(specs):
    ram = specs.get("ram_gb", 8)
    vram = specs.get("gpu_vram_gb", 0)
    if ram < 8: return "qwen3:1.7b"
    if ram <= 16 and vram < 8: return "ministral-3:3b"
    if vram >= 8 or ram > 24: return "ministral-3:8b"
    if vram >= 12: return "ministral-3:14b"
    return "llama3.1:8b"

def manage_log_files(log_dir: Path, max_logs: int):
    log_dir.mkdir(exist_ok=True)
    log_files = sorted(log_dir.glob('*.txt'), key=os.path.getmtime)
    while len(log_files) >= max_logs:
        os.remove(log_files[0])
        log_files.pop(0)

def optimize_image_bytes(image_bytes):
    """
    Attempts to convert image to WebP.
    Returns: (bytes, extension_string)
    """
    try:
        # Try to Optimize to WebP
        image = Image.open(io.BytesIO(image_bytes))
        
        # Handle Transparency (RGBA -> RGB/White Background)
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, (0, 0), image.convert("RGBA"))
            image = background
        else:
            image = image.convert("RGB")
        
        # Resize if massive
        if image.width > 1000 or image.height > 1000:
            image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
            
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='WEBP', quality=80, method=6)
        
        # SUCCESS: Return WebP data and "webp" extension
        return byte_arr.getvalue(), "webp"

    except Exception as e:
        print(f"Optimization warning: {e}. Reverting to original format.")
        
        # FAILURE: Detect original format so we don't save a PNG as .webp
        try:
            original = Image.open(io.BytesIO(image_bytes))
            ext = original.format.lower() if original.format else "png"
            return image_bytes, ext
        except:
            # Total failure: Default to png safe mode
            return image_bytes, "png"

def clear_cache(pdf_cache_dir: Path, ai_cache_dir: Path) -> str:
    results = []
    for name, cache_dir in [("PDF", pdf_cache_dir), ("AI", ai_cache_dir)]:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                results.append(f"✅ {name} cache cleared.")
            except Exception as e:
                results.append(f"❌ Error clearing {name} cache: {e}")
        cache_dir.mkdir(exist_ok=True)
    return " ".join(results)

def clean_filename_heuristics(raw_stem: str) -> str:
    # 1. Standard noise removal (keep this part)
    clean = re.sub(r'[_\.,]', ' ', raw_stem)
    clean = re.sub(r'^\s*(?:L|Winter|win|Spring|Lecture|Lec|S|Ssn|Session|Sessions|Week|Wk|MICRG|DOPOD|PHARM|Mod|Module)\s*\d+[\s:-]*', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'(?i)(pdf|ppt|pptx)$', '', clean)
    
    # 2. NEW: Remove specific academic noise words (keep this list or expand)
    noise_words = ['canvas', 'canva', 'final', 'exam', 'midterm', 'quiz', 'dopod', 'download', 'upload', 'version', 'draft', 'copy', 'review', 'ssn', 'session', 'chapter', 'chap', 'AZCOM']
    pattern = r'\b(' + '|'.join(noise_words) + r')\b'
    clean = re.sub(pattern, '', clean, flags=re.IGNORECASE)

    # 3. STRICT RULE: Allow ONLY Letters, Spaces, '&' and '+'
    # This removes all numbers (0-9) and all other punctuation
    clean = re.sub(r'[^a-zA-Z&+\s]', ' ', clean)

    # 4. Clean up whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    if clean.isupper(): clean = clean.title()
    return clean

def update_decks_from_files(files, max_decks: int):
    updates = []
    num_files = len(files) if files else 0
    
    for i in range(max_decks):
        visible = i < num_files
        deck_title_str = ""
        header_str = f"**Deck {i+1}**" # Default Header
        file_value = None
        
        if visible:
            path_str = files[i] 
            path_obj = Path(path_str)
            clean_name = clean_filename_heuristics(path_obj.stem)
            
            deck_title_str = clean_name
            
            # [OLD] Markdown version
            # header_str = f"### Deck {i+1}: {path_obj.name}"
            
            # [NEW] HTML version to match UI
            header_str = f"<div style='background-color: #228b22; color: white; padding: 4px; text-align: center; border-radius: 5px; font-weight: bold;'>Deck {i+1}: {path_obj.name}</div>"
            
            file_value = [path_str] 
            
        updates.extend([
            gr.update(visible=visible),                # 1. Update Group visibility
            gr.update(value=header_str),               # 2. Update Header Text
            gr.update(value=deck_title_str),           # 3. Update Title value
            gr.update(value=file_value, visible=False) # 4. Update File (Hidden)
        ])
    return updates

# Updated to accept a string path directly
def guess_lecture_details(file_path: str) -> str:
    if not file_path: return ""
    file_stem = Path(file_path).stem
    clean_name = clean_filename_heuristics(file_stem)
    return clean_name if clean_name else file_stem

# Fixed: Accepts list of strings (paths), removes recursive label update

import networkx as nx
import numpy as np
from sentence_transformers import util, CrossEncoder

# Cache models to prevent reloading
_cross_encoder_model = None
import torch
import networkx as nx
import numpy as np
from sentence_transformers import util, CrossEncoder

_cross_encoder_model = None

def rank_facts_with_cross_encoder(facts, objectives, threshold=0.15):
    """
    Calibrated Sieve (Shifted Sigmoid).
    Converts raw academic logits into human-readable percentages.
    Formula: 1 / (1 + e^-(x + 4))
    """
    global _cross_encoder_model
    
    if not facts or not objectives:
        return []

    # 1. Load Model
    if _cross_encoder_model is None:
        try:
            print("   > [System] Loading Cross-Encoder (ms-marco-MiniLM-L-12-v2)...")
            _cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        except Exception as e:
            print(f"   > [Error] Failed to load Cross-Encoder: {e}")
            return facts

    # 2. Build Pairs
    pairs = []
    for fact in facts:
        for obj in objectives:
            pairs.append([obj, fact])

    # 3. Predict (Get Raw Logits)
    raw_scores = _cross_encoder_model.predict(pairs)
    
    # 4. THE FORMULA: Shifted Sigmoid
    # We shift the logits by +4.0 to center the academic distribution.
    # Now, a raw score of -4.0 becomes 0.0 (50%), which makes sense for this data.
    calibrated_logits = torch.tensor(raw_scores) + 6.0
    probs = torch.sigmoid(calibrated_logits).numpy()

    # 5. Filter
    num_objs = len(objectives)
    high_yield_facts = []
    
    for i in range(len(facts)):
        start_idx = i * num_objs
        end_idx = start_idx + num_objs
        
        # Get the BEST probability this fact had against ANY objective
        fact_probs = probs[start_idx : end_idx]
        best_prob = np.max(fact_probs)
        
        # 6. Intuitive Thresholding
        # Now you can set threshold to 0.50 and it actually means "Average Confidence"
        if best_prob >= threshold:
            high_yield_facts.append(facts[i])

    print(f"   > [Sieve] Kept {len(high_yield_facts)}/{len(facts)} facts (Calibrated Prob >= {threshold})")
    
    return high_yield_facts

SETTINGS_FILE = Path("settings.json")

def save_settings(settings_dict: dict):
    try:
        with SETTINGS_FILE.open('w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with SETTINGS_FILE.open('r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def is_superfluous(text: str) -> bool:
    if not text: return True
    text_lower = text.strip().lower()
    if 'http://' in text_lower or 'https://' in text_lower:
        return True
    if len(text.split()) < 3:
        return True
    return False


def save_sieve_report(deck_name, all_facts, selected_facts):
    """
    Writes a report of which facts were kept vs dropped.
    """
    # 1. Determine what was dropped
    # We use the unique 'fact' text to identify them
    selected_texts = set(f['fact'] for f in selected_facts)
    dropped_facts = [f for f in all_facts if f['fact'] not in selected_texts]
    
    # 2. Prepare the content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"SIEVE REPORT: {deck_name}",
        f"Date: {timestamp}",
        f"Total Facts Harvested: {len(all_facts)}",
        f"Facts Kept: {len(selected_facts)}",
        f"Facts Dropped: {len(dropped_facts)}",
        "=" * 60,
        "",
        "--- SELECTED FACTS (High Yield) ---"
    ]
    
    for i, f in enumerate(selected_facts, 1):
        lines.append(f"{i}. [Pg {f.get('page_num', '?')}] {f['fact']}")
        
    lines.append("")
    lines.append("--- EXCLUDED FACTS (Low Yield / Tangential) ---")
    
    for i, f in enumerate(dropped_facts, 1):
        lines.append(f"{i}. [Pg {f.get('page_num', '?')}] {f['fact']}")

    # 3. Write to file
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    safe_name = "".join([c for c in deck_name if c.isalnum() or c in (' ', '-', '_')]).strip()
    filename = log_dir / f"Sieve_Report_{safe_name}_{int(datetime.now().timestamp())}.txt"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return f"Report saved: {filename}"
    except Exception as e:
        return f"Failed to save report: {e}"

# [Add to utils.py]


def get_ollama_models():
    """Fetches list of available models from local Ollama instance."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except:
        pass
    return []

def pull_ollama_model(model_name, progress=gr.Progress()):
    """Runs ollama pull in a subprocess."""
    if not model_name: return "No model selected."
    
    # --- CRITICAL FIX START ---
    # The UI sends "gemma3:1b | 815MB | Light"
    # We need to split by "|" and take the first part
    if "|" in model_name:
        clean_name = model_name.split("|")[0].strip()
    else:
        clean_name = model_name.strip()
    # --- CRITICAL FIX END ---

    # Check if Ollama is running
    try:
        requests.get("http://localhost:11434", timeout=1)
    except:
        return "❌ Error: Ollama is not running. Please open the Ollama app first."

    def run_pull():
        try:
            print(f"Starting download for {clean_name}...")
            # Use shell=True for better compatibility on Windows
            subprocess.run(f"ollama pull {clean_name}", shell=True, check=True)
            print(f"Finished download for {clean_name}!")
        except Exception as e:
            print(f"Pull failed: {e}")

    # Start in background thread so UI doesn't freeze
    thread = threading.Thread(target=run_pull)
    thread.start()
    
    return f"⬇️ Downloading {clean_name}... (Check your terminal/console for progress)"

def get_smart_recommendations(specs):
    """Returns a list of compatible models based on VRAM/RAM."""
    # 1. Database of Models (User Specified)
    # Sizes estimated: <1GB, ~2GB, ~4GB, ~6GB, ~8GB, ~10GB+
    # FORMAT: (Name, Size_Label, Min_RAM_GB, Description)
    ALL_MODELS = [
        ("gemma3:270m", "292MB", 1, "Ultra-Light "),
        ("qwen3:0.6b", "400MB", 1, "Ultra-Light "),
        ("gemma3:1b", "815MB", 2, "Light "),
        ("qwen3:1.7b", "1.5GB", 4, "Light "),
        ("gemma3:4b", "3.3GB", 6, "Balanced "),
        ("qwen3:4b", "3.0GB", 6, "Balanced "),
        ("emma3:latest", "3.3GB", 6, "Balanced "),
        ("gemma3n:e2b", "5.6GB", 8, "Mid-Range "),
        ("qwen3:8b", "5.5GB", 8, "Mid-Range "),
        ("gemma3n:e4b", "7.5GB", 12, "High-End "),
        ("gemma3:12b", "8.1GB", 16, "High-End "),
        ("qwen3:14b", "10GB", 16, "Ultra ")
    ]

    # 2. Determine Capacity
    # If NVIDIA GPU exists, use VRAM. If Mac/CPU, use System RAM (minus 4GB overhead)
    capacity = specs.get('gpu_vram_gb', 0)
    if not specs.get('has_nvidia', False):
         # For Mac/CPU, leave room for OS
        capacity = max(1, specs.get('ram_gb', 8) - 4) 

    # 3. Filter
    recommended = []
    for name, size, min_ram, desc in ALL_MODELS:
        if capacity >= min_ram:
            recommended.append((name, size, desc))
            
    # Always return at least the smallest ones if nothing fits
    if not recommended:
        recommended = [ALL_MODELS[0], ALL_MODELS[1]]
        
    return recommended