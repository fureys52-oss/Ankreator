import io
import base64
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import datetime
from datetime import datetime
from utils import optimize_image_bytes

def _optimize_image(image_bytes: bytes) -> Optional[bytes]:
    """Optimizes and standardizes an image to WebP format."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Keep this block! It converts transparent backgrounds to white.
        # This prevents black text from disappearing in Anki Night Mode.
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, (0, 0), image.convert("RGBA"))
            image = background
        
        image = image.convert("RGB")
        
        if image.width > 1000 or image.height > 1000:
            image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
            
        byte_arr = io.BytesIO()
        
        # CHANGE: Save as WEBP with varying quality
        image.save(byte_arr, format='WEBP', quality=80, method=6) # method=6 is max compression effort
        
        return byte_arr.getvalue()
    except Exception as e:
        print(f"Error optimizing image: {e}")
        return None

class ImageSource(ABC):
    """Abstract base class for image-finding strategy."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def search(self, query_text: str, clip_model: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        pass

class PDFImageSource(ImageSource):
    """Strategy to find PDF visuals using Hard-Gated Spatial Priority."""
    def __init__(self):
        super().__init__(name="PDF (AI Validated)")

    def search(self, query_text: str, clip_model: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        if not clip_model: 
            return None

        visual_inventory = kwargs.get("visual_inventory")
        source_pages = kwargs.get("full_source_page_numbers", [])
        
        if not visual_inventory: return None

        # Now it is safe to call .get()
        model = clip_model.get('clip') or clip_model.get('model')
        if not model: return None

        # Encode Query
        query_vec = model.encode([query_text])[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0: query_vec = query_vec / query_norm

        # --- 1. SEGMENT AND SCORE ---
        tier_data = {1: [], 2: [], 3: []}
        
        adjacent_pages = set()
        for p in source_pages:
            adjacent_pages.add(p - 1)
            adjacent_pages.add(p + 1)

        for item in visual_inventory:
            if 'embedding' not in item: continue
            
            sim = float(np.dot(item['embedding'], query_vec))
            item_page = item.get("page_num", -999)
            
            entry = {"score": sim, "visual": item}
            
            if item_page in source_pages:
                tier_data[1].append(entry)
            elif item_page in adjacent_pages:
                tier_data[2].append(entry)
            else:
                tier_data[3].append(entry)

        # --- 2. HIERARCHICAL WATERFALL CHECK ---
        # TIER 1: Same Page (Strict)
        if tier_data[1]:
            best = max(tier_data[1], key=lambda x: x["score"])
            if best["score"] >= 0.05:
                return {"image_bytes": best['visual']['image_bytes'], "source": self.name, "score": best['score']}

        # TIER 2: Adjacent Page (Medium)
        if tier_data[2]:
            best = max(tier_data[2], key=lambda x: x["score"])
            if best["score"] >= 0.20:
                return {"image_bytes": best['visual']['image_bytes'], "source": self.name, "score": best['score']}

        # TIER 3: Global (High Threshold)
        if tier_data[3]:
            best = max(tier_data[3], key=lambda x: x["score"])
            if best["score"] >= 0.35:
                return {"image_bytes": best['visual']['image_bytes'], "source": self.name, "score": best['score']}

        return None

class ImageFinder:
    """Orchestrator that runs the PDF search strategy."""
    def __init__(self):
        self.strategy = PDFImageSource()

    def find_best_image(self, query_texts: List[str], clip_model: Dict, **kwargs) -> Optional[str]:
        # Only use the first query text for PDF search
        pdf_visual_inventory = kwargs.get("pdf_visual_inventory", [])
        
        if pdf_visual_inventory:
            best_result = self.strategy.search(
                query_texts[0], 
                clip_model, 
                visual_inventory=pdf_visual_inventory,
                full_source_page_numbers=kwargs.get("full_source_page_numbers", [])
            )
            
            if best_result and best_result.get("image_bytes"):
                # [FIX] Handle tuple return here too
                optimized_bytes, ext = optimize_image_bytes(best_result["image_bytes"])
                
                if optimized_bytes:
                    b64_image = base64.b64encode(optimized_bytes).decode('utf-8')
                    timestamp = int(datetime.now().timestamp() * 1000)
                    
                    # [FIX] Use dynamic extension
                    filename = f"ADG_{timestamp}.{ext}"
                    
                    return {
                        "type": "upload_ready",
                        "data": b64_image,
                        "filename": filename,
                        "html": f'<img src="{filename}">'
                    }
            return None