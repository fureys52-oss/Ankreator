import json
import re
import io
import base64
import gc
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import fitz # PyMuPDF
from PIL import Image
import numpy as np
import cv2
import gradio as gr
from sentence_transformers import SentenceTransformer
import pymupdf4llm

from utils import configure_tesseract, is_superfluous, save_settings, optimize_image_bytes
from image_finder import ImageFinder, PDFImageSource
from prompts import (
    BUILDER_PROMPT, CLOZE_BUILDER_PROMPT, 
    CRITIC_PROMPT_TEMPLATE, GOLDEN_OBJECTIVE_PROMPT
)
import base64
from datetime import datetime
from llm_service import LLMService
import easyocr
OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
import warnings

# [Targeted Change: Silence PyTorch CPU warnings]
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")


def clean_anki_content(text):
    """Cleans common LLM artifacts and HTML mess."""
    if not text: return ""
    
    # 1. Remove Markdown Code Block Wrappers
    text = re.sub(r'^```(?:html|xml|markdown)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    
    # 2. Remove "Here is the card" conversational filler
    text = re.sub(r'(?i)^(here is|sure,|i can|created).{0,50}:\s*', '', text)
    
    # [CRITICAL FIX] Convert HTML breaks to real newlines
    text = re.sub(r'<br\s*/?>', '\n', text)
    
    # 3. Consolidate excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4. FORCE BULLETS (Standardize Markers to •)
    # Note: We capture spaces (\1) to preserve indentation for the styler
    text = re.sub(r'(?m)^(\s*)[-–—−⁃*](?!\d)', r'\1•', text)
    
    return text.strip()

# --- Constants ---
ANKI_CONNECT_URL = "http://127.0.0.1:8765"
MERMAID_JS_SCRIPT = """<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script><script>(function(){var el=document.getElementById("mermaid-{side}-{card_id}");if(el){var graph=el.textContent.trim();el.innerHTML=graph;mermaid.initialize({startOnLoad:false,theme:'MERMAID_THEME_PLACEHOLDER'});mermaid.run({nodes:[el]});}})();</script>"""

# --- Function Tools (Definitions) ---
CLOZE_COMPONENTS_TOOL = {
    "name": "create_cloze_components",
    "description": "Provides components for a cloze card.",
    "parameters": {
        "type": "object",
        "properties": {
            "Context_Question": {"type": "string"},
            "Full_Sentence": {"type": "string"},
            "Cloze_Keywords": {"type": "array", "items": {"type": "string"}},
            "Source_Page": {"type": "string"},
            "Search_Query": {"type": "string"},
            "Simple_Search_Query": {"type": "string"}
        },
        "required": ["Context_Question", "Full_Sentence", "Cloze_Keywords", "Source_Page", "Search_Query", "Simple_Search_Query"]
    }
}

NOTE_TYPE_CONFIG = {
    "basic": {
    "modelName": "ADG - Basic",
    "fields": ["Front", "Back", "Image", "Source"],
    "css": """.card { 
               font-family: Arial, sans-serif; 
               font-size: 20px; 
               text-align: center; 
               color: #333; 
               background-color: transparent;
             }
             .nightMode .card { color: #f0f0f0; }

             /* 0. The Text Wrapper (Back of Card) */
             /* Keeps text centered on screen but left-aligned */
             .content-wrapper {
               display: block;
               max-width: 1200px; 
               margin: 0 auto; 
               text-align: left; /* FORCE LEFT ALIGNMENT FOR TEXT */
             }

             /* 3. Image Styling */
             img { 
               /* Inline-block allows them to sit in a row */
               display: inline-block; 
               
               /* The 300px Hard Limit */
               height: 300px !important; 
               width: auto; 
               
               /* Spacing */
               margin: 5px; 
               
               object-fit: contain;
               border-radius: 5px; 
               box-shadow: 0 4px 6px rgba(0,0,0,0.1);
               vertical-align: middle; 
             }

             /* --- TIGHTER SPACING RULES --- */
             
             /* Reduce vertical space around the whole list */
             ul { 
               list-style-type: none;
               padding-left: 0; 
               margin-top: 0.2em; 
               margin-bottom: 0.2em;
             }
             
             /* Remove extra space for nested lists */
             ul ul {
                margin-top: 0; 
                margin-bottom: 0;
             }

             /* Tighten individual lines */
             li { 
               position: relative;
               padding-left: 1.5em;
               margin-bottom: 0.25em; 
               line-height: 1.3;
             }
             
             /* Prevent paragraphs from adding huge gaps */
             p {
                margin: 0.2em 0;
             }

             /* Custom Bullet Points */
             li::before { 
               content: '•';
               position: absolute; 
               left: 0.3em; 
               top: 0;
               font-size: 1.2em;
             }
             
             ul ul li::before {
               content: '○';
               font-size: 1.0em;
               top: 0.1em;
             }
          """,
    "templates": [ 
        { 
            "Name": "Card 1", 
            "Front": """{{Front}}""", 
            # FIX IS HERE: Removed <div class="content-wrapper"> from around {{FrontSide}}
            # Now {{FrontSide}} stays centered, but {{Back}} is wrapped and goes left.
            "Back": """{{FrontSide}}<hr id=answer><div class="content-wrapper">{{Back}}</div><br><br>{{#Image}}{{Image}}{{/Image}}<div style='font-size:12px; color:grey;'>{{Source}}</div>""" 
        } 
    ],
    "function_tool": { 
        "name": "create_anki_card", 
        "description": "Creates a single Anki card based on a conceptual chunk of facts.", 
        "parameters": { 
            "type": "object", 
            "properties": { 
                "Topic": {
                    "type": "string", 
                    "description": "The main subject (e.g. 'Aortic Stenosis')."
                },
                "Sub_Topics": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Exhaustive list of specific aspects covered (e.g. ['Mechanism', 'Symptoms'])."
                },
                # --- NEW FIELD ---
                "Exhaustive_Question": {
                    "type": "string",
                    "description": "A comprehensive, natural-language question that explicitly asks for the information related to the Sub_Topics. Do not just list them."
                },
                "Back": {"type": "string"}, 
                "Page_numbers": {"type": "array", "items": {"type": "integer"}}, 
                "Search_Query": {"type": "string"}, 
                "Simple_Search_Query": {"type": "string"} 
            }, 
            "required": ["Topic", "Sub_Topics", "Exhaustive_Question", "Back", "Page_numbers", "Search_Query", "Simple_Search_Query"] 
        } 
    }
},
    "cloze": {
        "modelName": "ADG - Atomic Cloze", "fields": ["Text", "Extra", "Image", "Source"], "isCloze": True,
        "css": ".card { font-family: Arial; font-size: 20px; text-align: center; } .cloze { font-weight: bold; color: %s; } img { max-height: 500px; min-width: 400px; min-height: 250px; object-fit: contain; }",
        "templates": [ { "Name": "Cloze Card", "Front": "{{Extra}}<br><br>{{cloze:Text}}", "Back": """{{Extra}}<br><br>{{cloze:Text}}<br><br>{{#Image}}{{Image}}{{/Image}}<div style='font-size:12px; color:grey;'>{{Source}}</div>""" } ],
        "function_tool": CLOZE_COMPONENTS_TOOL
    },
    "mermaid": {
        "modelName": "ADG - Mermaid Diagram", "fields": ["Front", "Back", "MermaidFront", "MermaidBack", "Source", "Image"],
        "css": ".card { font-family: Arial; font-size: 20px; text-align: center; } .mermaid { background-color: #f9f9f9; padding: 10px; border-radius: 5px; }",
        "templates": [{"Name": "Card 1", "Front": "{{Front}}" + f"<div id='mermaid-front-{{{{CardID}}}}' class='mermaid'>{{{{MermaidFront}}}}</div>" + MERMAID_JS_SCRIPT.replace("{side}", "front").replace("{card_id}", "{{CardID}}"), "Back": "{{Back}}" + f"<div id='mermaid-back-{{{{CardID}}}}' class='mermaid'>{{{{MermaidBack}}}}</div>" + "{{#Image}}{{Image}}{{/Image}}<div style='font-size:12px; color:grey;'>{{Source}}</div>" + MERMAID_JS_SCRIPT.replace("{side}", "back").replace("{card_id}", "{{CardID}}")}],
        "function_tool": { "name": "create_mermaid_card", "description": "Creates a Mermaid card.", "parameters": {"type": "object", "properties": {"Front": {"type": "string"}, "Back": {"type": "string"}, "Mermaid_Front_Code": {"type": "string"}, "Mermaid_Back_Code": {"type": "string"}, "Page_numbers": {"type": "array", "items": {"type": "integer"}}, "Search_Query": {"type": "string"}, "Simple_Search_Query": {"type": "string"}}, "required": ["Front", "Back", "Mermaid_Front_Code", "Mermaid_Back_Code", "Page_numbers", "Search_Query", "Simple_Search_Query"]}}
    }
}

class CardSurgeon:
    """
    Structure-Aware Splitter with Hybrid Cost Logic.
    Calculates cost based on the GREATER of Word Count or Structural Complexity.
    Preserves parent-child hierarchy during splits.
    """
    def __init__(self, settings_dict):
        # The User's "Max Facts" slider is the target budget for a single card
        self.target_budget = float(settings_dict.get("max_facts_input", 5))
        
        # Cost Settings
        self.words_per_fact = float(settings_dict.get("surgeon_word_limit", 20))
        self.bullet_cost = float(settings_dict.get("surgeon_bullet_cost", 1.0))
        self.sub_bullet_cost = float(settings_dict.get("surgeon_sub_bullet_cost", 0.5))

    def _calculate_block_cost(self, parent_line, child_lines):
        """
        Calculates the 'Fact Equivalent' cost of a block.
        Logic: Max(Word_Based_Cost, Structure_Based_Cost)
        """
        # 1. Word Based Cost
        full_text = parent_line + " " + " ".join(child_lines)
        word_count = len(full_text.split())
        word_cost = word_count / self.words_per_fact

        # 2. Structure Based Cost
        # Base cost for the parent bullet
        structure_cost = self.bullet_cost
        
        # Analyze Children (The Exception Rule)
        # Exception: 3+ children that are short (1-3 words) count as "Lists", not full facts
        is_simple_list = len(child_lines) >= 3 and all(len(c.split()) <= 3 for c in child_lines)
        
        for child in child_lines:
            if is_simple_list:
                # Discounted cost for simple lists
                structure_cost += (self.sub_bullet_cost * 0.5)
            else:
                structure_cost += self.sub_bullet_cost

        # Return whichever is higher (User said: "whichever is larger")
        return max(word_cost, structure_cost)

    def operate(self, original_back_text):
        lines = original_back_text.split('\n')
        if not lines: return [[original_back_text]]

        # --- STEP 1: PARSE INTO LOGICAL BLOCKS ---
        blocks = [] # List of tuples: (parent_string, [child_strings])
        current_parent = None
        current_children = []

        for line in lines:
            if not line.strip(): continue
            
            # Detect Hierarchy
            stripped = line.strip()
            # Indentation check: < 2 spaces is Parent, >= 2 spaces is Child
            leading_spaces = len(line) - len(line.lstrip())
            is_parent = leading_spaces < 2
            
            if is_parent:
                # Save previous block
                if current_parent is not None:
                    blocks.append((current_parent, current_children))
                
                # Start new block
                current_parent = line
                current_children = []
            else:
                # It's a child (or continuation), attach to current parent
                if current_parent is None:
                    # Orphan child (rare), treat as parent
                    current_parent = line
                else:
                    current_children.append(line)
        
        # Save last block
        if current_parent is not None:
            blocks.append((current_parent, current_children))

        # --- STEP 2: BIN PACKING (Based on Cost) ---
        bins = []
        current_bin = []
        current_bin_cost = 0.0
        
        for parent, children in blocks:
            # Reconstruct string for output
            full_block_str = parent + ("\n" + "\n".join(children) if children else "")
            
            # Calculate Cost
            cost = self._calculate_block_cost(parent, children)
            
            # Decision: Add to current bin or start new one?
            # We allow a small overflow (10%) to prevent stranding a single bullet
            if current_bin and (current_bin_cost + cost > self.target_budget * 1.1):
                bins.append(current_bin)
                current_bin = [full_block_str]
                current_bin_cost = cost
            else:
                current_bin.append(full_block_str)
                current_bin_cost += cost
        
        if current_bin:
            bins.append(current_bin)
            
        return bins

def apply_intelligent_styling(text: str, color_map: Dict[str, str], context_text: str = "", dynamic_dict: Dict = None) -> str:
    """
    Layered Heuristic Highlighter v8.0: Token-Stream Logic.
    Goes word-by-word to ensure stability and correct clustering.
    """
    if not isinstance(text, str) or not text: return ""

    # --- PALETTE ---
    C_HEADER   = "#E0F2FE"
    C_STRUCT   = color_map.get('structure') or "#38BDF8"
    C_TOPIC    = color_map.get('topic')     or "#FDBA74"
    C_DATA     = color_map.get('data')      or "#2DD4BF"
    C_ANATOMY  = color_map.get('anatomy')   or "#60A5FA"
    C_PHARMA   = color_map.get('pharma')    or "#F472B6"
    C_PROCESS  = color_map.get('process')   or "#A78BFA"
    C_POS      = color_map.get('pos')       or "#38BDF8"
    C_NEG      = color_map.get('neg')       or "#F87171"
    C_BOLD     = "#F8B680" 
    C_ITALIC   = "#FEF08A" 
    C_SCENARIO = "#94A3B8" 

    # --- COMPILE DICTIONARIES ---
    # We compile a list of (Pattern, Color) tuples to iterate over
    rules = []

    # 1. Dynamic
    if dynamic_dict:
        if dynamic_dict.get('anatomy'):
            rules.append((r'\b(' + '|'.join([re.escape(t) for t in dynamic_dict['anatomy']]) + r')\b', C_ANATOMY))
        if dynamic_dict.get('pathology'):
            rules.append((r'\b(' + '|'.join([re.escape(t) for t in dynamic_dict['pathology']]) + r')\b', C_NEG))
        if dynamic_dict.get('pharmacology'):
            rules.append((r'\b(' + '|'.join([re.escape(t) for t in dynamic_dict['pharmacology']]) + r')\b', C_PHARMA))
        if dynamic_dict.get('physiology'):
            rules.append((r'\b(' + '|'.join([re.escape(t) for t in dynamic_dict['physiology']]) + r')\b', C_POS))

    # 2. Context (Topic)
    if context_text:
        stop_words = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'about', 'as', 'into', 'like', 'through', 'after', 'over', 'between', 'out', 'against', 'during', 'without', 'before', 'under', 'around', 'among', 'and', 'but', 'or', 'so', 'yet', 'because', 'although', 'since', 'unless', 'while', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'explain', 'describe', 'list', 'define', 'identify', 'compare', 'contrast', 'discuss', 'outline', 'diagram', 'image', 'picture', 'figure', 'table', 'show', 'illustrate', 'key', 'main', 'primary', 'type', 'types', 'what', 'why', 'how', 'when', 'where', 'specifically', 'regarding'}
        clean_ctx = re.sub(r'[^\w\s]', '', context_text.lower())
        context_words = [w for w in clean_ctx.split() if w not in stop_words and len(w) > 3]
        if context_words:
            # Match strictly whole words
            pattern = r'\b(' + '|'.join([re.escape(w) for w in context_words]) + r')\b'
            rules.append((pattern, C_TOPIC))
    if dynamic_dict:
            if dynamic_dict.get('anatomy'):
                rules.append((r'\b(' + '|'.join([re.escape(t) for t in dynamic_dict['anatomy']]) + r')\b', color_map.get('anatomy') or "#60A5FA"))
            if dynamic_dict.get('pathology'):
                rules.append((r'\b(' + '|'.join([re.escape(t) for t in dynamic_dict['pathology']]) + r')\b', color_map.get('neg') or "#F87171"))
            if dynamic_dict.get('drugs'):
                rules.append((r'\b(' + '|'.join([re.escape(t) for t in dynamic_dict['drugs']]) + r')\b', color_map.get('pharma') or "#F472B6"))
        # 3. Static Regexes (Prioritized Order)
    # Quantitative
    rules.append((r'(?<!\w)(?:>|<|≥|≤|~|approx\.?|±)?\s*\d+(?:\.\d+)?(?:[-/]\d+(?:\.\d+)?)?\s*(?:mg|g|kg|ml|L|dL|cm|mm|µm|nm|mmHg|kPa|%|°C|F|µg|IU|Eq|mEq|mol|mmol|hr|min|sec|wk|yr|days?|wks?|yrs?|types?|stage|grade)\b|\b(exponential|logarithmic|binary|double[sd]?|triple[sd]?|quadruple[sd]?|phase|cycle|ratio|frequency|density|count|volume|prevalence|incidence|mortality|morbidity|mean|median|mode|range|p-value|confidence|sensitivity|specificity)\b', C_DATA))
    
    # Pharma
    rules.append((r'\b\w+(?:imab|zumab|umab|cillin|cyclin|mycin|micin|azole|vir|olol|pril|sartan|pine|ide|sone|lone|prazole|afil|barbital|caine|triptan|glinide|glitazone|dronate|parin|mab|nib|floxacin|penem|thromycin|cycline|navir|vudine)\b', C_PHARMA))
    rules.append((r'\b(?:cef|ceph|sulfa|nitro|pred|cort)\w*\b', C_PHARMA))
    rules.append((r'\b(drug|medication|agent|antibiotic|antiviral|antifungal|analgesic|sedative|anesthetic|diuretic|steroid|nsaid|statin|vaccine|antitoxin|antidote|placebo|prophylaxis|treatment|therapy|regimen|dosage|dose|pill|tablet|capsule|injection|infusion|intravenous|oral|topical|sublingual|intramuscular|subcutaneous)\w*\b', C_PHARMA))

    # Pathology (Negative)
    rules.append((r'\b(decreased?|low|deficien(t|cy)|loss|failure|damage|death|injury|trauma|severe|acute|chronic|impair(ed|ment)|inhibit(s|ion|ed)?|block(s|ed)?|destr(oy|uction)|hypo\w*|suppress(ed|ion)?|compromise(d)?|worsen(ed|ing)?|exacerbat(e|ed|ion)|atrophy|necrosis|ischemia|obstruction|malignant|tumor|cancer|lesion|syndrome|disorder|disease|pathology|abnormal|mutation|defect|risk|complication|inflam(mation|matory)|infection|sepsis|shock|edema|fibrosis|stenosis|sclerosis|hernia|ulcer|abscess|cyst|fistula|prolapse|reflux|spasm|palsy|paralysis|seizure|stroke|infarct|toxin|toxic|poison|exhaust(ion|ed)?|deplet(ion|ed)|accumulat(ion|ed)?|stagnation|metastasis|relapse|recurrence|pain|tenderness|fever|pyrexia|cough|dyspnea|shortness of breath|sob|tachycardia|tachypnea|bradycardia|bradypnea|arrhythmia|murmur|rub|gallop|crackles|rales|wheeze|stridor|rash|erythema|cyanosis|jaundice|pallor|fatigue|malaise|nausea|vomiting|emesis|diarrhea|constipation|bleeding|hemorrhage|anemia|hypoxia|hypoxemia)\b', C_NEG))
    rules.append((r'\b\w+(?:itis|osis|pathy|oma|megaly|penia|emia|iasis|trophy|plasia|malacia|stasis|ptysis|rrhea|rrhexis|sclerosis|stenosis)\b', C_NEG))

    # Physiology (Positive)
    rules.append((r'\b(increased?|high|elevat(ed|ion)|gain|recover(y|ed)|heal(ed|ing)|normal|stable|stimulat(e|es|ion|ed)|activat(e|es|ion|ed)|hyper\w*|enhanc(e|ed|ement)|promot(e|es)|induc(e|es|tion)|potentiate(s|d)?|generat(e|ion)|develop(s|ed|ment)?|adapt(ation)?|compensat(e|ion)|reserve|perfusi(on|ed)|innervat(ion|ed)|viable|vital|growth|proliferat(ion|e)|active|progeny|homeostasis|equilibrium|balance|intact|resolution|remission|benign|protection|immunity|resistance|tolerance)\b', C_POS))

    # Anatomy/Micro
    rules.append((r'\b(lung|heart|liver|kidney|brain|stomach|bowel|colon|intestine|nerve|muscle|bone|joint|blood|vessel|artery|vein|capillary|lymph|skin|tissue|cell|membrane|receptor|ligand|neuron|axon|dendrite|myelin|synapse|ganglion|plexus|cortex|medulla|lobe|ventricle|atrium|aorta|valve|septum|pericardium|myocardium|endocardium|gastric|hepatic|cardiac|cerebral|spinal|thalamus|hypothalamus|pituitary|renal|nephron|glomerulus|tubule|calyx|ureter|bladder|urethra|esophagus|sphincter|pylorus|duodenum|jejunum|ileum|mucosa|submucosa|serosa|epithelium|endothelium|dermis|epidermis|keratin|collagen|elastin|cartilage|tendon|ligament|fascia|spleen|thymus|marrow|thyroid|adrenal|pancreas|biliary|gallbladder|retina|cornea|cochlea|vestibule|serum|plasma|urine|stool|feces|sputum|saliva|csf|synovial|fluid|distal|proximal|lateral|medial|anterior|posterior|dorsal|ventral|superior|inferior|nutrient)\w*\b', C_ANATOMY))
    rules.append((r'\b(bacteria|virus|fungus|parasite|protozoa|helminth|prion|pathogen|organism|flora|mitochondria|nucleus|ribosome|lysosome|endoplasmic|reticulum|golgi|cytoplasm|cytoskeleton|cilia|flagella|pili|fimbriae|capsule|spore|plasmid|capsid|envelope|glycocalyx|biofilm|antigen|antibody|macrophage|neutrophil|leukocyte|lymphocyte|platelet|culture|medium|media|agar|broth|colony|sample|specimen|biomass|turbidity|swab|biopsy|smear)\w*\b', C_ANATOMY))

    # Process
    rules.append((r'\b(synthesis|production|metabolism|secretion|excretion|absorption|digestion|replication|transcription|translat(ion|ed)|recycling|mechanism|pathway|cascade|feedback|transport|bind(s|ing)?|cleav(e|age)|signal(ing)?|regulat(e|ion)|convert(s|ed)?|catalyz(e|ed)|phosphorylat(e|ion)|function|interact(ion)?|express(ion)?|diffus(ion|ed)|osmosis|filtration|conduction|contraction|relaxation|divis(ion|ided)|fission|inoculat(ion|ed)|stain(ing|ed)?|plot(ted|ting)?|measur(ed|ement)|determin(ed|ing)|uptake|release|storage|differentiation|maturation|apoptosis|mitosis|meiosis|cloning|sequencing|pcr|electrophoresis)\b', C_PROCESS))
    rules.append((r'\b\w+(?:ase|ose|in|ine|ate|ol|yl)\b', C_PROCESS))

    # --- PROCESSING LOOP (Line by Line) ---
    lines = text.split('\n')
    output_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            output_lines.append("<br>")
            continue

        # --- INDENTATION LOGIC ---
        # Calculate raw spaces at start of line
        leading_spaces = len(line) - len(line.lstrip())
        
        # Determine Nesting Level (Simpler Logic)
        if leading_spaces < 2: indent_level = 0
        elif leading_spaces < 5: indent_level = 1
        else: indent_level = 2
        
        # --- LIST MARKER DETECTION ---
        clean_stripped = stripped.replace(u'\u200b', '') 
        list_match = re.match(r'^(?:(?:\\*\\*|\'\')?)([-*+•–—−⁃])(?:(?:\\*\\*|\'\')?)\\s*(.*)', clean_stripped)
        
        is_list = False
        list_marker = ""
        content = stripped
        
        if list_match and not re.match(r'^-\d', stripped):
            marker_char = list_match.group(1)
            content = list_match.group(2)
            
            # Visual Marker Style
            list_marker = "•" if indent_level == 0 else ("◦" if indent_level == 1 else "▪")
            is_list = True
            
        elif re.match(r'^\d+\.\s+', stripped):
            match = re.match(r'^(\d+\.)\s+(.*)', stripped)
            list_marker = match.group(1)
            content = match.group(2)
            is_list = True

        # 2. Check for Headers (#)
        if re.match(r'^#', content):
            clean = content.replace("#", "").strip()
            output_lines.append(f'<h3 style="color: {C_HEADER}; margin: 15px 0 5px 0; border-bottom: 1px solid #555; padding-bottom: 2px;">{clean}</h3>')
            continue

        # 3. Check for Structural Key (The Colon Rule)
        prefix_html = ""
        processing_text = content
        
        key_match = re.match(r'^([^\:]{1,100}):\s+(.*)', content)
        if key_match:
            potential_key = key_match.group(1)
            rest_of_line = key_match.group(2)
            clean_key = re.sub(r'[*_]', '', potential_key).strip()
            
            if len(clean_key.split()) <= 2:
                prefix_html = f'<strong style="color: {C_STRUCT};">{clean_key}</strong>:&nbsp;'
                processing_text = rest_of_line

        # 4. Tokenization & Coloring (Keep existing logic)
        tokens = re.split(r'(\s+|[.,;!?()\[\]])', processing_text)
        token_data = []
        for t in tokens:
            if not t: continue
            is_sep = bool(re.match(r'^(\s+|[.,;!?()\[\]])$', t))
            token_data.append({'text': t, 'color': None, 'is_sep': is_sep})

        # Apply Rules (Keep existing loop)
        for t in token_data:
            if t['is_sep']: continue
            clean_t = re.sub(r'[*_]', '', t['text'])
            for pattern, color in rules:
                if re.search(pattern, clean_t, re.IGNORECASE):
                    t['color'] = color
                    break
            if '**' in t['text']: t['color'] = C_TOPIC

        # Domino/Clustering Logic (Keep existing loop)
        current_chain_color = None
        for i in range(len(token_data)):
            t = token_data[i]
            if re.match(r'^[.,;!?]$', t['text']):
                current_chain_color = None; continue
            if t['is_sep']: continue
            if t['color']:
                if current_chain_color: t['color'] = current_chain_color
                else: current_chain_color = t['color']
            else: current_chain_color = None

        # Parenthesis Logic (Keep existing loop)
        for i in range(len(token_data)):
            t = token_data[i]
            if not t['color']: continue
            next_idx = i + 1
            while next_idx < len(token_data) and re.match(r'^\s+$', token_data[next_idx]['text']): next_idx += 1
            if next_idx < len(token_data) and token_data[next_idx]['text'] == '(':
                color = t['color']
                token_data[next_idx]['color'] = color
                scan = next_idx + 1
                while scan < len(token_data):
                    token_data[scan]['color'] = color
                    if token_data[scan]['text'] == ')': break
                    scan += 1

        # HTML Reconstruction (Keep existing logic)
        final_line_html = prefix_html
        buffer_text = ""; buffer_color = None
        for t in token_data:
            if t['color'] == buffer_color: buffer_text += t['text']
            else:
                if buffer_text:
                    display = buffer_text.replace('**', '')
                    weight = "bold" if '**' in buffer_text else "normal"
                    if buffer_color: final_line_html += f'<span style="color: {buffer_color}; font-weight: {weight};">{display}</span>'
                    elif '**' in buffer_text: final_line_html += f'<b>{display}</b>'
                    else: final_line_html += buffer_text
                buffer_text = t['text']; buffer_color = t['color']
        
        if buffer_text:
            display = buffer_text.replace('**', '')
            weight = "bold" if '**' in buffer_text else "normal"
            if buffer_color: final_line_html += f'<span style="color: {buffer_color}; font-weight: {weight};">{display}</span>'
            elif '**' in buffer_text: final_line_html += f'<b>{display}</b>'
            else: final_line_html += buffer_text

        final_line_html = re.sub(r'\*(.*?)\*', f'<span style="color: {C_ITALIC};">\\1</span>', final_line_html)

        # Output Construction
        margin = indent_level * 30 
        
        if is_list:
            output_lines.append(f'<div style="margin-left: {margin}px; margin-bottom: 4px; line-height: 1.4;"><span style="color: {C_STRUCT}; margin-right: 8px; font-weight: bold; font-size: 1.2em;">{list_marker}</span>{final_line_html}</div>')
        else:
            # Even plain text follows indentation if provided
            output_lines.append(f'<div style="margin-left: {margin}px; margin-bottom: 8px; line-height: 1.4;">{final_line_html}</div>')

    return "".join(output_lines)

def invoke_ankiconnect(action: str, **params: Any) -> tuple[Optional[Any], Optional[str]]:
    payload = json.dumps({"action": action, "version": 6, "params": params}, ensure_ascii=False)
    try:
        response = requests.post(ANKI_CONNECT_URL, data=payload.encode('utf-8'), headers={'Content-Type': 'application/json'}, timeout=30)
        response.raise_for_status()
        result = response.json()
        return (result.get('result'), result.get('error'))
    except Exception as e:
        return None, f"AnkiConnect Error: {e}"

def upload_media_to_anki(filename, b64_data):
    """Uploads a base64 string to Anki's media collection."""
    params = {
        "filename": filename,
        "data": b64_data
    }
    return invoke_ankiconnect("storeMediaFile", **params)

def setup_anki_deck_and_note_type(deck_name: str, note_type_key: str, cloze_color: str, mermaid_theme: str) -> Optional[str]:
    config = NOTE_TYPE_CONFIG.get(note_type_key)
    if not config: return "Config Error"
    _, err = invoke_ankiconnect("createDeck", deck=deck_name)
    if err: return err

    final_css = config.get("css", "") % cloze_color if "%s" in config.get("css", "") else config.get("css", "")
    final_templates = []
    for t in config.get("templates", []):
        new_t = t.copy()
        new_t["Front"] = t["Front"].replace("MERMAID_THEME_PLACEHOLDER", mermaid_theme)
        new_t["Back"] = t["Back"].replace("MERMAID_THEME_PLACEHOLDER", mermaid_theme)
        final_templates.append(new_t)

    params = {
        "modelName": config["modelName"], "inOrderFields": config["fields"], "css": final_css,
        "isCloze": config.get("isCloze", False), "cardTemplates": final_templates
    }
    model_names, _ = invoke_ankiconnect("modelNames")
    if config["modelName"] not in model_names:
        _, err = invoke_ankiconnect("createModel", **params)
    else:
        invoke_ankiconnect("updateModelStyling", model={"name": config["modelName"], "css": final_css})
        invoke_ankiconnect("updateModelTemplates", model={"name": config["modelName"], "templates": {t['Name']: {'Front': t['Front'], 'Back': t['Back']} for t in final_templates}})
    return err

def add_note_to_anki(deck_name, note_type_key, fields, tags):
    config = NOTE_TYPE_CONFIG[note_type_key]
    note = {"deckName": deck_name, "modelName": config["modelName"], "fields": fields, "options": {"allowDuplicate": False}, "tags": tags}
    return invoke_ankiconnect("addNote", note=note)

class DeckProcessor:
    def __init__(self, deck_name, files, logger, progress, all_settings, cache_dirs, clip_model, excluded_indices):
        self.deck_name = deck_name
        self.files = files
        self.pdf_paths = []
        if files:
            for f in files:
                if isinstance(f, str):
                    self.pdf_paths.append(f)
                elif hasattr(f, 'name'):
                    self.pdf_paths.append(f.name)
        self.log = logger
        self.progress = progress
        self.pdf_cache_dir, self.ai_cache_dir = cache_dirs
        self.clip_model = clip_model.get('model') if clip_model else None
        self.excluded_indices = excluded_indices or []
        
        # Store settings
        self.all_settings = all_settings
        
        # Parse logic settings
        self.content_strategy = all_settings.get("content_strategy", "Extract All Facts")
        self.objectives_text_manual = all_settings.get("objectives_text_manual", "")
        self.enable_vision_extraction = all_settings.get("enable_vision_extraction", False)
        
        # Parse dynamic lists
        self.dynamic_terms = {
            'anatomy': [t.strip() for t in all_settings.get('style_anatomy', '').split(',') if t.strip()],
            'drugs': [t.strip() for t in all_settings.get('style_drugs', '').split(',') if t.strip()],
            'pathology': [t.strip() for t in all_settings.get('style_pathology', '').split(',') if t.strip()],
        }

        # Setup standard vars
        # --- FIX: Detect Card Type from Settings instead of hardcoding ---
        mode = all_settings.get("card_generation_mode", "Basic (QA Bullets)")
        if "Cloze" in mode:
            self.card_type = ["Atomic Cloze"]
        else:
            self.card_type = ["Basic"]
            
        self.custom_tags = [t.strip() for t in all_settings.get("custom_tags", "").split(',') if t.strip()]
        self.cloze_color = all_settings.get("cloze_color", "#3b82f6")
        self.mermaid_theme = all_settings.get("mermaid_theme", "default")
        self.color_map = {k: all_settings.get(f"c_{k}") for k in ['structure', 'topic', 'data', 'anatomy', 'pharma', 'process', 'pos', 'neg']}
        self.note_type_key = "basic"
        self.llm_service = None
        self.pdf_visual_inventory = []
        self.image_finder = None
        self.golden_objectives = []
        self.full_text = ""

        self.PHASE_WEIGHTS = {
            "setup": 0.05,      # 5% for text extraction/curation
            "visuals": 0.10,    # 10% for finding images
            "harvester": 0.30,  # 30% for finding facts (heavy API usage)
            "librarian": 0.05,  # 5% for cleaning
            "assembly": 0.05,   # 5% for clustering
            "creator": 0.40,    # 40% for writing cards (heaviest API usage)
            "export": 0.05      # 5% for sending to Anki
        }
        self.current_global_progress = 0.0

        reader = easyocr.Reader(['en'], gpu=False)

    def _initialize_ai(self):
        if not self.llm_service:
            # CHANGED: Instantiate LLMService with settings
            self.llm_service = LLMService(self.log, self.all_settings)

    def _setup_anki(self):
        # Determine Note Type
        if "Basic" in self.card_type: self.note_type_key = "basic"
        elif "Atomic Cloze" in self.card_type: self.note_type_key = "cloze"
        elif "Contextual Cloze" in self.card_type: self.note_type_key = "contextual_cloze"
        elif "Mermaid" in self.card_type: self.note_type_key = "mermaid"
        else: self.note_type_key = "basic"

        self.log(f"Card Type Selected: {self.card_type}")
        if self.custom_tags: self.log(f"Custom Tags: {', '.join(self.custom_tags)}")
        
        # Call Utils to create deck
        primary_error = setup_anki_deck_and_note_type(self.deck_name, self.note_type_key, self.cloze_color, self.mermaid_theme)
        
        if primary_error:
            self.log(f"DECK SETUP ERROR: {primary_error}")
            
            # --- NEW: User-Facing Popups ---
            err_msg = primary_error.lower()
            if "connection" in err_msg or "refused" in err_msg or "403" in err_msg:
                # This stops execution and shows a red popup
                raise gr.Error("🛑 Anki Connect Error: Ensure Anki is OPEN and the AnkiConnect add-on is installed.")
            else:
                raise gr.Error(f"🛑 Anki Error: {primary_error}")
            
            return False
            
        return True
    
    def _process_pdfs(self):
        self.log("\n--- Processing PDF Files (Dual-Mode: Markdown & Raw) ---")
        self.full_text = ""      # Raw Text (Perfect for Regex/Scout)
        self.full_markdown = ""  # Markdown (Perfect for LLM/Harvester)
        
        for pdf_path in self.pdf_paths:
            try:
                doc = fitz.open(pdf_path)
                total_pages = len(doc) # <--- HARD LIMIT
                
                # 1. VISUAL EXCLUSIONS
                all_indices = list(range(len(doc)))
                keep_indices = [i for i in all_indices if i not in self.excluded_indices]
                
                if not keep_indices:
                    self.log(f"   > Skipping {Path(pdf_path).name} (All pages excluded)")
                    continue

                # 2. MARKDOWN CONVERSION
                # This sometimes returns more items than pages (the "phantom page" issue)
                md_data = pymupdf4llm.to_markdown(doc, pages=keep_indices, page_chunks=True)
                
                file_buffer_md = f"\n\n# Source: {Path(pdf_path).name}\n"
                file_buffer_raw = f"\n\n# Source: {Path(pdf_path).name}\n"
                
                for entry in md_data:
                    # 'page' metadata is the original 0-based index
                    original_idx = entry['metadata']['page']
                    p_num = original_idx + 1
                    
                    # A. Build Markdown Stream (Safe)
                    file_buffer_md += f"\n--- Page {p_num} ---\n{entry['text']}\n"
                    
                    # B. Build Raw Text Stream (Preventative Logic)
                    # We ONLY call load_page if the index is real. 
                    if 0 <= original_idx < total_pages:
                        try:
                            raw_page_content = doc.load_page(original_idx).get_text("text")
                        except Exception:
                            # If individual page load fails for any other reason, fallback
                            raw_page_content = entry['text']
                    else:
                        # This handles the "Markdown has one more page" case.
                        # We use the text existing in the markdown entry instead of crashing.
                        self.log(f"   > [System] Skipped raw load for phantom page index {original_idx} (Doc len: {total_pages})")
                        raw_page_content = entry['text']

                    file_buffer_raw += f"\n--- Page {p_num} ---\n{raw_page_content}\n"
                
                self.full_markdown += file_buffer_md
                self.full_text += file_buffer_raw

            except Exception as e:
                # Catches file-level corruption, not page-level logic errors
                raise gr.Error(f"🛑 Could not process '{Path(pdf_path).name}'.\nError: {e}")
                
        self.log("Text extraction complete (Markdown + Raw).")
        return True

    def _curate_text_pages(self) -> str:
        if not self.all_settings.get("auto_curation", True):
            self.log("   > Auto-Curation disabled by user. Using all pages.")
            return self.full_text

        self.log("\n--- Running Python-Based Page Curation ---")
        MIN_CHARS_PER_PAGE, MAX_LINES_FOR_TITLE, MAX_LINE_LENGTH_FOR_TITLE = 50, 2, 250
        JUNK_PAGE_PATTERNS = [r'learning objectives',r'objectives',r'objective', r'table of contents', r'references', r'bibliography', r'acknowledgements', r'session log', r'Midwestern Wellness Support', r'@midwestern.edu']
        
        curated_text_parts, pages_kept, pages_dropped = [], 0, 0
        page_pattern = re.compile(r'(--- Page (\d+) ---\n(.*?)(?=(--- Page \d+ ---)|\Z))', re.DOTALL)
        matches = list(page_pattern.finditer(self.full_markdown))
        
        for match in matches:
            full_block, page_num, page_content = match.group(1), match.group(2), match.group(3).strip()
            
            if len(page_content) < MIN_CHARS_PER_PAGE:
                self.log(f"   > Dropping Page {page_num}: Low text content.")
                pages_dropped += 1; continue
            
            lines = [line for line in page_content.split('\n') if line.strip()]
            if lines and len(lines) < MAX_LINES_FOR_TITLE and max(len(line) for line in lines) < MAX_LINE_LENGTH_FOR_TITLE:
                self.log(f"   > Dropping Page {page_num}: Title page detected.")
                pages_dropped += 1; continue
            
            if any(re.search(pattern, page_content, re.IGNORECASE) for pattern in JUNK_PAGE_PATTERNS):
                self.log(f"   > Dropping Page {page_num}: Junk content detected.")
                pages_dropped += 1; continue
            
            curated_text_parts.append(full_block)
            pages_kept += 1
            
        final_text = "".join(curated_text_parts)
        if pages_kept == 0:
            self.log("   > CRITICAL: Curation removed all pages. Reverting to full text.")
            return self.full_text
        
        self.log(f"   > Curation complete. Kept {pages_kept}/{len(matches)} pages.")
        return final_text

    def _extract_headers_via_font(self, doc):
        headers = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            if s["size"] > 14:
                                headers.append(s["text"])
        return headers

    def _auto_extract_objectives(self, text_source) -> Optional[List[str]]:
        self.log("   > [Objectives] Running extraction...")
        objective_keywords = ['learning objectives', 'learning outcome', 'session objectives', 'session goals', 'learning goals', 'enabling objectives', 'objectives']
        relevant_text = ""
        
        scan_window = text_source[:10000]
        if any(k in scan_window.lower() for k in objective_keywords):
            relevant_text = scan_window
        else:
            self.log("   > [Objectives] Keywords failed. Using AI Scan on first 5 pages.")
            return self.llm_service.extract_objectives(scan_window)

        py_objs = self._python_extract_objectives(relevant_text)
        if py_objs:
            self.log(f"   > [Objectives] Python found {len(py_objs)} objectives.")
            return py_objs
            
        return self.llm_service.extract_objectives(relevant_text)

    def _python_extract_objectives(self, text_input) -> List[str]:
        """
        Robust Objective Extractor v5 (Stream-Based).
        - ELIMINATED BLOCK SPLITTING: Treats text as one continuous stream.
        - Fixes "Header in Block A, List in Block B" separation issues.
        - Uses a state machine to link Headers -> Lists across page gaps.
        """
        # --- 1. DEFENSIVE CASTING ---
        if text_input is None: return []
        text = ""
        if isinstance(text_input, list):
            text = "\n".join([str(x) for x in text_input if x])
        else:
            text = str(text_input)

        # --- 2. PATTERNS ---
        # Catch standard bullets, numbered lists (1.), and lettered lists (a.)
        item_pattern = re.compile(r'^\s*(?:(?:\d+|[a-zA-Z]|[IVX]+)[\.\)]|[-•●*➢⇒>])\s*(.*)', re.IGNORECASE)
        
        # Skip garbage lines (Emails, Page markers, University headers)
        junk_pattern = re.compile(r'^\s*(page\s*\d+|.*@.*\.\w+|midwestern|azcom|university|slide|www\.|http)', re.IGNORECASE)
        
        # Catch headers like "Learning Objectives" or just "Objectives"
        header_pattern = re.compile(r'(?:learning|session|enabling|chapter|module|lecture|unit)\s+(?:objectives|goals|outcomes|competencies|aims)|(?:\bobjectives\b)', re.IGNORECASE)

        lines = text.split('\n')
        
        clusters = []
        current_cluster = []
        
        # State Machine Flags
        header_found = False
        lines_since_header = 0     # How far are we from the 'Objectives' title?
        lines_since_last_item = 0  # How far are we from the last bullet point?
        
        # --- 3. STREAM PROCESSING ---
        for line in lines:
            clean = line.strip()
            if not clean: continue

            # EVENT A: Found a Header
            if header_pattern.search(clean):
                # If we were already building a list, save it and start fresh
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
                
                header_found = True
                lines_since_header = 0
                continue # Done with this line

            # If we haven't found a header yet, ignore everything
            if not header_found:
                continue

            # EVENT B: Junk Line (Skip without penalty)
            if junk_pattern.match(clean) or clean.startswith("--- Page"):
                continue

            # EVENT C: Found List Item
            m = item_pattern.match(clean)
            if m:
                content = m.group(1).strip()
                if len(content) > 4: # Filter "1." artifacts
                    current_cluster.append(content)
                    # Reset counters because we found valid content
                    lines_since_last_item = 0
                    lines_since_header = 0 
            
            # EVENT D: Text Line (Continuation or Gap)
            else:
                # Are we inside a list? (Continuation Logic)
                if current_cluster and lines_since_last_item < 3:
                    # Heuristic: Wrapped lines usually start lowercase or are long
                    if len(clean) > 50 or (not clean.endswith(':') and not clean[0].isupper()):
                         current_cluster[-1] += " " + clean
                         lines_since_last_item = 0 # It was part of the item
                    else:
                         lines_since_last_item += 1
                else:
                    lines_since_last_item += 1
                
                lines_since_header += 1

            # TERMINATION CONDITIONS
            # 1. We found a header, but saw 20 lines of text with NO list items. False alarm.
            if not current_cluster and lines_since_header > 20:
                header_found = False
                
            # 2. We were in a list, but saw 5 lines of non-list text. List ended.
            if current_cluster and lines_since_last_item > 5:
                clusters.append(current_cluster)
                current_cluster = []
                header_found = False 

        # Capture any final list at EOF
        if current_cluster:
            clusters.append(current_cluster)

        # --- 4. AGGREGATION & CLEANUP ---
        valid_candidates = [c for c in clusters if len(c) >= 2]
        if not valid_candidates: return []

        all_objectives = []
        for cluster in valid_candidates:
            all_objectives.extend(cluster)

        # Deduplicate
        seen = set()
        unique_objs = []
        for obj in all_objectives:
            clean_obj = obj.strip()
            if clean_obj not in seen:
                unique_objs.append(clean_obj)
                seen.add(clean_obj)
                
        return unique_objs

    def _run_scout_and_objectives(self, full_text):
        self.log("\n--- Phase 0: Scout & Objectives (Python Optimized) ---")
        
        # 1. Manual Override Check
        if self.objectives_text_manual:
             self.log("   > [Scout] Using provided manual objectives.")
             self.golden_objectives = [o.strip() for o in self.objectives_text_manual.split('\n') if o.strip()]
             return

        # 2. FAST HEADER EXTRACTION (Map the document structure)
        # We scan for lines that look like headers (short, distinct, larger font)
        headers = []
        try:
            doc = fitz.open(self.pdf_paths[0])
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    for l in b.get("lines", []):
                        for s in l.get("spans", []):
                            # Heuristic: Font size > 11 usually indicates a header in academic slides/papers
                            if s["size"] > 11 and len(s["text"].strip()) > 3: 
                                headers.append(s["text"].strip())
        except Exception: pass
        
        # Limit to top 50 unique headers to give the AI a 'Table of Contents'
        structure_context = "Document Structure:\n" + "\n".join(list(set(headers[:50])))

        # 3. TARGETED OBJECTIVE SEARCH (Bookend Pages)
        # This is where 99% of learning objectives live.
        self.log("   > [Scout] Scanning bookend pages for objectives...")
        
        total_pages = len(doc)
        # Scan First 5 and Last 5 pages
        target_indices = list(range(0, min(10, total_pages))) + list(range(max(0, total_pages-5), total_pages))
        target_indices = sorted(list(set(target_indices))) 
        
        relevant_text = ""
        for i in target_indices:
            relevant_text += doc[i].get_text() + "\n"

        found_objs = []

        # --- STRATEGY A: Explicit Headers (Regex) ---
        # Catches sections explicitly labeled "Objectives", "Goals", "Outcomes", etc.
        obj_patterns = [
            r'(?i)(?:(?:learning|session|enabling|chapter|module|lecture|unit)\s+(?:objectives|goals|outcomes|competencies|aims)|(?:\bobjectives\b))[:\s]*(.*?)(?=\n\n|\n[A-Z])',
            r'(?i)at the end of this (?:session|lecture|module|chapter|unit),? you (?:will|should) be able to:?\s*(.*?)(?=\n\n|\n[A-Z])',
            r'(?i)after completing this (?:unit|section),? students will:?\s*(.*?)(?=\n\n|\n[A-Z])',
            r'(?i)aims?:?\s*(.*?)(?=\n\n|\n[A-Z])',
            r'(?i)purpose:?\s*(.*?)(?=\n\n|\n[A-Z])',
            r'(?i)expected results?:?\s*(.*?)(?=\n\n|\n[A-Z])',
            r'(?i)key concepts?:?\s*(.*?)(?=\n\n|\n[A-Z])',
            r'(?i)competenc(?:y|ies):?\s*(.*?)(?=\n\n|\n[A-Z])'
        ]
        
        for pat in obj_patterns:
            matches = re.findall(pat, relevant_text, re.DOTALL)
            for m in matches:
                # Split multiline matches into individual bullet points
                lines = [line.strip().lstrip('-•*').strip() for line in m.split('\n') if len(line.strip()) > 5]
                found_objs.extend(lines)

        # --- STRATEGY B: Action Verb Discovery ---
        # Catches lines starting with strong academic verbs (e.g., "Define X", "Explain Y")
        action_verbs = [
            "Define", "Describe", "Explain", "List", "Identify", "State", "Name", "Label", "Recall", "Recognize", 
            "Summarize", "Outline", "Review", "Discuss", "Clarify", "Paraphrase", "Locate", "Select", "Match", "Cite", 
            "Apply", "Analyze", "Calculate", "Demonstrate", "Illustrate", "Interpret", "Use", "Differentiate", 
            "Distinguish", "Compare", "Contrast", "Categorize", "Classify", "Relate", "Correlate", "Examine", 
            "Investigate", "Prioritize", "Diagram", "Assess", "Evaluate", "Predict", "Justify", "Critique", "Formulate", 
            "Design", "Construct", "Propose", "Recommend", "Conclude", "Plan", "Verify", "Argue", "Diagnose", "Treat", 
            "Manage", "Prescribe", "Perform", "Order", "Interpret", "Palpate", "Auscultate", "Counsel", "Educate", 
            "Prevent", "Screen"
        ]
        
        # Regex: Start of line -> Optional bullet -> Verb -> Content
        verb_pattern = r'(?m)^\s*(?:[-•*0-9.]+\s+)?(' + '|'.join(action_verbs) + r')\s+(.*)'
        verb_matches = re.findall(verb_pattern, relevant_text, re.IGNORECASE)
        
        for verb, content in verb_matches:
            full_obj = f"{verb} {content}".strip()
            # Quality filter: Must be a reasonable sentence length (15-200 chars)
            if 15 < len(full_obj) < 200:
                found_objs.append(full_obj)

        # 4. THE GOLDENIZER STEP (Refine & Deduplicate)
        # Clean and sort unique findings
        unique_raw_objs = sorted(list(set(found_objs)), key=len, reverse=True)
        raw_count = len(unique_raw_objs)
        
        if raw_count > 0:
            self.log(f"   > [Scout] Python found {raw_count} raw potential objectives.")
            self.log(f"   > [Goldenizer] Refining raw findings into Golden Objectives...")
            
            # Combine headers (context) + raw findings (data)
            context_block = structure_context + "\n\nRaw Findings:\n" + "\n".join([f"- {o}" for o in unique_raw_objs])
            
            # Call AI to synthesize the final list
            self.golden_objectives = self.llm_service.run_goldenizer(context_block)
            self.log(f"   > [Goldenizer] Created {len(self.golden_objectives)} Golden Objectives.")
            
        else:
            self.log("   > [Scout] No raw objectives found. Analyzing headers...")
            # Fallback: Ask AI to infer objectives from the Headers we extracted
            prompt = f"Based on these headers from a lecture, list 8 key learning objectives:\n\n{structure_context}"
            try:
                response = self.llm_service._call_llm([{"role": "user", "content": prompt}])
                self.golden_objectives = [line.strip('- ').strip() for line in response.split('\n') if '-' in line][:8]
            except:
                self.golden_objectives = ["Understand the core concepts of the material."]
    
    def _build_visual_inventory_from_pdf(self, curated_pages_text: str):
        if not self.pdf_paths: return
        self.log(f"\n--- Building Visual Inventory (Input: {Path(self.pdf_paths[0]).name}) ---")
        
        # HEURISTICS
        MIN_DIM = 150
        MAX_ASPECT = 4.0
        MONO_THRESHOLD = 0.90
        TEXT_HEAVY_LIMIT = 400
        MAX_EMBEDDED_COVERAGE = 0.80
        MIN_EMBEDDED_COVERAGE = 0.04
        MIN_LAYOUT_WIDTH = 30
        MIN_LAYOUT_HEIGHT = 30
        EDGE_TO_EDGE_THRESHOLD = 0.95
        
        RENDER_MAX_TEXT_CHARS = 150
        RENDER_MIN_VECTOR_PATHS = 5

        try: doc = fitz.open(self.pdf_paths[0])
        except Exception as e: self.log(f"  > CRITICAL: Failed to open PDF for visuals. Error: {e}"); return
        
        # --- HELPER DEFINITION ---
        def is_valid_visual(img_bytes: bytes, rect: fitz.Rect, page_ref: fitz.Page, source_type: str) -> bool:
            try:
                # 1. Byte Size Check
                if len(img_bytes) < 2048: return False 
                
                pil_img = Image.open(io.BytesIO(img_bytes))
                w, h = pil_img.size
                
                # 2. Minimum Dimension Check
                if w < MIN_DIM or h < MIN_DIM: return False
                
                # 3. Aspect Ratio Check
                ratio = w / h if h > 0 else 0
                if ratio > MAX_ASPECT or ratio < (1/MAX_ASPECT): return False
                
                # 4. Edge-to-Edge / Background Check
                if rect and page_ref:
                    page_area = page_ref.rect.get_area()
                    img_area = rect.get_area()
                    if page_area > 0 and (img_area / page_area) > EDGE_TO_EDGE_THRESHOLD:
                        return False

                if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
                
                # 5. Solid Color Block Check
                colors = pil_img.getcolors(maxcolors=1024) 
                if colors:
                    top_color = sorted(colors, key=lambda x: x[0], reverse=True)[0]
                    pixel_count = w * h
                    if (top_color[0] / pixel_count) > MONO_THRESHOLD: return False 
                
                # 6. Text-Heavy Check
                if source_type == "rendered" or (w > 300 and h > 200):
                    if rect and page_ref:
                        text_in_area = page_ref.get_text("text", clip=rect)
                        if len(text_in_area.strip()) > TEXT_HEAVY_LIMIT:
                            return False
                return True
            except Exception: 
                return False

        # --- PHASE 1: EMBEDDED IMAGES ---
        pages_with_good_embedded_images = set()
        self.log("\n--- Phase 1: Scanning for embedded images ---")
        
        for page in doc:
            if (page.number) in self.excluded_indices: continue
            found_on_page = False
            page_area = page.rect.get_area()
            image_list = page.get_images(full=False)
            
            for img_info in image_list: 
                xref_id = img_info[0]
                primary_rect = None
                is_structural_junk = False
                
                try:
                    rects = page.get_image_rects(xref_id)
                    if rects:
                        # Grab the first rect for validation context
                        primary_rect = rects[0]
                        for r in rects:
                            if page_area > 0 and (r.get_area() / page_area) > MAX_EMBEDDED_COVERAGE: is_structural_junk = True; break
                            if page_area > 0 and (r.get_area() / page_area) < MIN_EMBEDDED_COVERAGE: is_structural_junk = True; break
                            if r.width < MIN_LAYOUT_WIDTH or r.height < MIN_LAYOUT_HEIGHT: is_structural_junk = True; break
                except: pass 
                
                if is_structural_junk: continue

                try:
                    base_image = doc.extract_image(xref_id)
                    image_bytes = base_image["image"]
                    
                    # [FIX] PASSED ALL 4 REQUIRED ARGUMENTS
                    if is_valid_visual(image_bytes, primary_rect, page, "embedded"):
                        self.pdf_visual_inventory.append({
                            "image_bytes": image_bytes, 
                            "context_text": page.get_text("text") or " ", 
                            "page_num": page.number + 1, 
                            "source_method": "embedded"
                        })
                        found_on_page = True
                except Exception as e: 
                    # Log error to see if something else breaks
                    # print(f"Embedded extraction error: {e}") 
                    continue
                    
            if found_on_page: pages_with_good_embedded_images.add(page.number + 1)
            
        self.log(f"  > Phase 1 Complete: Found embedded images on {len(pages_with_good_embedded_images)} page(s).")

        # --- PHASE 2: RENDERED PAGE SECTIONS ---
        curated_page_nums = {int(p) for p in re.findall(r'--- Page (\d+) ---', curated_pages_text)}
        pages_to_render = sorted([p for p in curated_page_nums if p not in pages_with_good_embedded_images])

        if pages_to_render:
            self.log(f"\n--- Phase 2: Rendering {len(pages_to_render)} pages (Strict Filter) ---")
            DETECTION_DPI = 100
            EXTRACTION_DPI = 300
            
            for page_num in pages_to_render:
                try:
                    if page_num - 1 >= len(doc): continue
                    page = doc.load_page(page_num - 1)
                    if len(page.get_text("text").strip()) > RENDER_MAX_TEXT_CHARS: continue
                    if len(page.get_drawings()) < RENDER_MIN_VECTOR_PATHS: continue

                    pix = page.get_pixmap(matrix=fitz.Matrix(DETECTION_DPI / 72, DETECTION_DPI / 72))
                    img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4: cv_image = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
                    else: cv_image = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                    
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                    text_mask = np.zeros_like(gray)
                    for b in page.get_text("blocks"):
                        x0, y0, x1, y1 = [v * (DETECTION_DPI / 72) for v in b[:4]]
                        cv2.rectangle(text_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
                        
                    final_mask = cv2.subtract(thresh, text_mask)
                    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w < 50 or h < 50: continue
                        pdf_rect = fitz.Rect(x * (72/DETECTION_DPI), y * (72/DETECTION_DPI), (x+w)*(72/DETECTION_DPI), (y+h)*(72/DETECTION_DPI))
                        high_res_pix = page.get_pixmap(matrix=fitz.Matrix(EXTRACTION_DPI/72, EXTRACTION_DPI/72), clip=pdf_rect)
                        candidate_bytes = high_res_pix.tobytes("png")
                        
                        # [FIX] PASSED ALL 4 REQUIRED ARGUMENTS
                        if is_valid_visual(candidate_bytes, pdf_rect, page, "rendered"):
                            self.pdf_visual_inventory.append({
                                "image_bytes": candidate_bytes, 
                                "context_text": page.get_text("text") or " ", 
                                "page_num": page_num, 
                                "source_method": "rendered"
                            })
                except: pass
                    
        
        if self.pdf_visual_inventory and self.clip_model and isinstance(self.clip_model, dict):
            clip_model = self.clip_model.get('clip')
            if clip_model:
                self.log(f"   > Pre-computing AI embeddings for {len(self.pdf_visual_inventory)} visuals...")
                try:
                    pil_images = []; valid_indices = []
                    for i, item in enumerate(self.pdf_visual_inventory):
                        try:
                            img = Image.open(io.BytesIO(item['image_bytes'])).convert("RGB")
                            img = img.resize((224, 224))
                            pil_images.append(img); valid_indices.append(i)
                        except: continue
                    if pil_images:
                        image_embeddings = clip_model.encode(pil_images, batch_size=32)
                        for i, vector in zip(valid_indices, image_embeddings):
                            norm = np.linalg.norm(vector)
                            self.pdf_visual_inventory[i]['embedding'] = vector / norm if norm > 0 else vector
                    
                    self.log("   > [System] Unloading CLIP model to free RAM...")
                    del self.clip_model['clip']
                    self.clip_model['clip'] = None
                    gc.collect()
                except Exception as e: self.log(f"   > Warning: Embedding pre-computation failed: {e}")

        self.log(f"--- Visual inventory complete. Found {len(self.pdf_visual_inventory)} total visual assets. ---")

    def _construct_cloze_card_python(self, data):
        text = data.get("prose_summary", "")
        targets = data.get("targets", [])
        targets = list(set(targets))
        matches = []
        text_lower = text.lower()
        
        for target in targets:
            if not target: continue
            try:
                pattern = re.escape(target.lower())
                for m in re.finditer(pattern, text_lower):
                    start, end = m.span()
                    matches.append({"actual": text[start:end], "start": start, "end": end})
            except: continue
        
        matches.sort(key=lambda x: x["start"], reverse=True)
        unique_matches = []
        if matches:
            current = matches[0]
            unique_matches.append(current)
            for m in matches[1:]:
                if m['end'] <= current['start']:
                    unique_matches.append(m)
                    current = m

        for i, match in enumerate(unique_matches):
            cloze_id = (i // 2) + 1
            text = text[:match['start']] + f"{{{{c{cloze_id}::{match['actual']}}}}}" + text[match['end']:]
            
        return {"Header": data.get("header", "Concept"), "Text": text}

    def _validate_and_repair_mermaid_code(self, code):
        if not code: return ""
        code = code.replace("```mermaid", "").replace("```", "").strip()
        return code

    def _is_bundle_a_list(self, group_facts):
        if len(group_facts) < 4: return False
        total_words = sum(len(f['fact'].split()) for f in group_facts)
        avg_words = total_words / len(group_facts)
        return avg_words < 7

    def _run_critic_agent(self, topic, old_question, new_back):
        return self.llm_service.run_critic(topic, old_question, new_back)

    def _add_notes_to_anki(self, final_cards: List[Dict]):
        needs_reload = False
        if self.clip_model is None:
            needs_reload = True
        elif isinstance(self.clip_model, dict) and self.clip_model.get('clip') is None:
            needs_reload = True
        
        if needs_reload:
            self.log("   > [System] Reloading CLIP model for Image Search...")
            try:
                from sentence_transformers import SentenceTransformer
                # [CRITICAL FIX] Re-initialize dict if it was None
                if self.clip_model is None: self.clip_model = {}
                self.clip_model['clip'] = SentenceTransformer('clip-ViT-B-32', device='cpu')
            except Exception as e: self.log(f"   > Error reloading CLIP: {e}")

        self.log(f"\n--- Adding Cards to Deck: '{self.deck_name}' ---")
        # ... (rest of the function remains the same)
        cards_added, cards_skipped, cards_failed = 0, 0, 0
        main_pdf_path = self.pdf_paths[0] if self.pdf_paths else "Unknown.pdf"
        MAX_IMAGES_PER_CARD = 3
        
        for card_data in final_cards:
            try:
                image_html = None
                forced_visuals = []
                
                # --- 1. VISION EXTRACTION (Local Images from PDF) ---
                # This only runs if you CHECKED the button in the UI
                if self.enable_vision_extraction:
                    target_pages = card_data.get("args", {}).get("Page_numbers", [])
                    if not target_pages: target_pages = card_data.get("Page_numbers", [])
                    
                    all_candidates = []
                    for p_num in target_pages:
                        page_images = [x for x in self.pdf_visual_inventory if x['page_num'] == p_num]
                        all_candidates.extend(page_images)
                    
                    if all_candidates:
                        self.log(f"   > [Visuals] Found {len(all_candidates)} candidates on source pages {target_pages}")
                        def get_image_area(img_entry):
                            try:
                                with Image.open(io.BytesIO(img_entry['image_bytes'])) as img:
                                    return img.width * img.height
                            except: return 0
                        
                        unique_candidates = {id(x): x for x in all_candidates}.values() 
                        sorted_candidates = sorted(unique_candidates, key=get_image_area, reverse=True)
                        final_selection = sorted_candidates[:MAX_IMAGES_PER_CARD]

                        for i, match in enumerate(final_selection):
                            try:
                                # [FIX] Get both data AND extension
                                raw_bytes = match['image_bytes']
                                opt_bytes, ext = optimize_image_bytes(raw_bytes) # <--- Unpack tuple
                                
                                b64_data = base64.b64encode(opt_bytes).decode('utf-8')
                                timestamp = int(datetime.now().timestamp() * 1000)
                                # [FIX] Use the dynamic extension
                                filename = f"ADG_Vision_{timestamp}_{i}.{ext}"
                                
                                upload_media_to_anki(filename, b64_data)
                                
                                img_tag = f'<img src="{filename}">'
                                if img_tag not in forced_visuals: forced_visuals.append(img_tag)
                            except Exception as e:
                                self.log(f"   > [Visuals] Upload failed: {e}")
                
                # --- 2. IMAGE FINDER (Search / Fallback) ---
                if forced_visuals:
                    image_html = " ".join(forced_visuals)
                    self.log("   > [Visuals] Applied Tier 1 (Same Page) Image.")
                
                elif self.image_finder:
                    query_text = card_data.get("args", {}).get("Search_Query") or card_data.get("Search_Query")
                    if query_text:
                        # Call Finder
                        image_result = self.image_finder.find_best_image(
                            query_texts=[query_text], 
                            clip_model=self.clip_model, 
                            
                            # CRITICAL FIX: The variable name is 'pdf_visual_inventory'
                            pdf_visual_inventory=self.pdf_visual_inventory,
                            
                            full_source_page_numbers=card_data.get("args", {}).get("Page_numbers", [])
                        )
                        
                        if image_result:
                            # Handle Dict (New Upload Method)
                            if isinstance(image_result, dict) and "data" in image_result and "filename" in image_result:
                                upload_media_to_anki(image_result['filename'], image_result['data'])
                                image_html = image_result['html']
                                self.log(f"   > [Visuals] Applied Tier 2/3 Image search for '{query_text}'.")
                            
                            # Fallback (Old String Method - just in case)
                            elif isinstance(image_result, str):
                                image_html = image_result

                # --- FIELDS & EXPORT ---
                # Safe access to Page_numbers
                p_nums = card_data.get("args", {}).get("Page_numbers", [])
                if not p_nums: p_nums = [1]
                page_str = f"Pgs {', '.join(map(str, sorted(list(set(p_nums)))))}"
                source_text = f"{Path(main_pdf_path).stem} - {page_str}"
                fields, final_note_type_key = {}, None

                if card_data.get('name') == 'create_anki_card':
                    args = card_data['args']
                    final_note_type_key = 'basic'
                    
                    # [MODIFIED BLOCK START] ------------------------
                    
                    # 1. Clean the raw text FIRST
                    clean_front = clean_anki_content(args["Front"])
                    clean_back_source = clean_anki_content(args["Back"])

                    # 2. Pass the CLEAN text to the styler
                    back_html = apply_intelligent_styling(
                        clean_back_source, 
                        self.color_map, 
                        dynamic_dict=self.dynamic_terms
                    )
                    
                    fields = { 
                        "Front": clean_front, 
                        "Back": back_html, 
                        "Source": source_text, 
                        "Image": image_html or "" 
                    }
                    # [MODIFIED BLOCK END] --------------------------
                    pass
                elif card_data.get('name') == 'create_cloze_card':
                    args = card_data['args']
                    
                    # 1. Clean Inputs
                    raw_text = args.get("Prose") or args.get("Text", "")
                    keywords = args.get("Keywords", [])
                    header_text = args.get("Topic") or args.get("Header", "")
                    
                    # 2. Apply Cloze Syntax {{c1::...}}
                    if keywords and raw_text:
                        mode = self.all_settings.get("cloze_syntax_mode", "Combined")
                        grp_size = int(self.all_settings.get("cloze_group_size", 1))
                        final_text = self._apply_cloze_syntax(raw_text, keywords, mode, grp_size)
                    else:
                        # Fallback: If no keywords, just use text. 
                        # WARNING: If this has no {{c1::}} tags, Anki will reject it.
                        final_text = raw_text 

                    # 3. Clean Content
                    final_text = clean_anki_content(final_text)
                    header_text = clean_anki_content(header_text)
                    
                    # [CRITICAL FIX]
                    # Always use 'cloze' key if the function is create_cloze_card
                    final_note_type_key = 'cloze' 
                    
                    # 4. Map Fields (Matching your Config exactly)
                    fields = {
                        "Text": final_text, 
                        "Extra": header_text, 
                        "Source": source_text, 
                        "Image": image_html or ""
                    }
                    
                    # Safety Check: Does the text actually have a cloze?
                    if "{{" not in final_text:
                        self.log(f"   > Warning: Cloze card generated without cloze syntax. Skipping.")
                        cards_skipped += 1
                        continue

                if not fields: 
                    cards_skipped += 1; continue

                _, error = add_note_to_anki(self.deck_name, final_note_type_key, fields, self.custom_tags)
                if error:
                    if "duplicate" in error: cards_skipped += 1
                    else: self.log(f"   > FAILED to add note. Reason: {error}"); cards_failed += 1
                else: cards_added += 1

            except Exception as e:
                cards_failed += 1; self.log(f"ERROR processing card data: {e}")
                import traceback
                traceback.print_exc()

        self.log(f"\n--- Final Tally ---\nCards Added: {cards_added}\nCards Skipped/Failed: {cards_skipped + cards_failed}")

    def _apply_cloze_syntax(self, text, keywords, mode="Combined", group_size=1):
        if not keywords or not text: return text
        
        # Sort by length to prevent partial matches (e.g. matching "cell" inside "cellulose")
        keywords = sorted(list(set(keywords)), key=len, reverse=True)
        final_text = text
        
        # 1. Tokenize: Replace keywords with temporary placeholders to avoid double-processing
        # We store the keyword text in a map so we can restore it later
        kw_map = {}
        for i, kw in enumerate(keywords):
            token = f"__CLOZE_TOKEN_{i}__"
            kw_map[token] = kw
            # Case-insensitive replacement
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            final_text = pattern.sub(token, final_text)

        # 2. Apply Logic: Restore tokens with specific {{c::}} indices
        for i, token in enumerate(kw_map.keys()):
            original_word = kw_map[token]
            
            # --- THE LOGIC CORE ---
            if "Combined" in mode:
                c_id = 1
            elif "Sequential" in mode:
                c_id = i + 1
            elif "Grouped" in mode:
                # e.g. Size 2: 0,1->c1 | 2,3->c2
                c_id = (i // int(group_size)) + 1
            elif "Alternating" in mode:
                # e.g. Size 2: 0->c1, 1->c2, 2->c1, 3->c2
                c_id = (i % int(group_size)) + 1
            else:
                c_id = 1 # Fallback
            # ----------------------

            replacement = f"{{{{c{c_id}::{original_word}}}}}"
            final_text = final_text.replace(token, replacement)
            
        return final_text

    def _update_global_bar(self, phase_key, item_current, item_total, message):
        """
        Calculates global progress:
        Global = (Sum of previous phases) + (Current Phase Weight * (item_current / item_total))
        """
        # 1. Sum up completed phases
        base_progress = 0.0
        found_current = False
        
        for key, weight in self.PHASE_WEIGHTS.items():
            if key == phase_key:
                found_current = True
                break
            base_progress += weight
            
        # 2. Add fraction of current phase
        if item_total > 0:
            phase_progress = (item_current / item_total) * self.PHASE_WEIGHTS.get(phase_key, 0)
        else:
            phase_progress = 0
            
        total = min(base_progress + phase_progress, 0.99) # Cap at 99% until done
        
        # 3. Update Gradio (The magic happens here)
        self.progress(total, desc=f"[{self.deck_name}] {message}")

    def run(self):
        self.log(f"--- Starting Processing for Deck: {self.deck_name} ---")
        self._initialize_ai()
        if not self._setup_anki(): return None
        if not self._process_pdfs(): return None

        # --- PHASE 1: SCOUT ---
        self._build_visual_inventory_from_pdf(self.full_markdown)
        
        # 1. Check Strategy explicitly
        if "Objective" in self.content_strategy:
             self.log("   > [Scout] Strategy includes Objectives. Checking source...")
             
             raw_objs = []
             
             # A. MANUAL OVERRIDE (The missing piece)
             if "Provided" in self.content_strategy and self.objectives_text_manual:
                 self.log("   > [Scout] Using Manual Objectives provided in UI.")
                 # Split by newline and filter empty lines
                 raw_objs = [line.strip() for line in self.objectives_text_manual.split('\n') if len(line.strip()) > 3]
                 
             # B. AUTO-EXTRACTION (Regex)
             else:
                 self.log("   > [Scout] Scanning text for 'Learning Objectives' headers...")
                 raw_objs = self._python_extract_objectives(self.full_text)
             
             # 2. Refine & Store (Goldenizer)
             if raw_objs:
                 origin = "Manual Input" if "Provided" in self.content_strategy else "Regex"
                 self.log(f"   > [Scout] Found {len(raw_objs)} candidates via {origin}. Refining...")
                 
                 # We send even manual objectives to the Goldenizer to clean/deduplicate them
                 self.golden_objectives = self.llm_service.run_goldenizer("\n".join(raw_objs))
                 self.log(f"   > [Goldenizer] Finalized {len(self.golden_objectives)} Golden Objectives.")
             else:
                 # 3. Explicit Failure Log
                 self.log("   > [Scout] ⚠️ No objectives found. (Check headers or UI input).")
                 self.golden_objectives = []
        else:
             self.log("   > [Scout] Objective extraction skipped (Strategy is 'Extract All Facts').")
        # --- CURATION STEP ---
        # Uses the actual curation logic to remove junk pages
        if self.all_settings.get("auto_curation", True):
            self.log("--- Curation Step (Active) ---")
            harvest_input = self._curate_text_pages()
        else:
            harvest_input = self.full_markdown

        # --- PHASE 2: HARVESTER ---
        self._update_global_bar("harvester", 0, 1, "Harvesting...")
        raw_facts = self.llm_service.run_markdown_harvester(harvest_input)
        
        if not raw_facts:
            self.log("❌ Harvester found no facts. Aborting.")
            return None

        # --- OPTIMIZATION: SIEVE BEFORE LIBRARIAN ---
        # We filter FIRST so we don't waste time "cleaning" irrelevant facts.
        
        sieved_facts = raw_facts

        # Only run if we have objectives to filter against
        if "Objective" in self.content_strategy and self.golden_objectives:
                    
                    # Check if Filtering is enabled
            use_sieve = self.all_settings.get("use_ppr_filtering", True)
            
            if use_sieve:
                self.log(f"   > [Sieve] Running Cross-Encoder Sieve...")
                from utils import rank_facts_with_cross_encoder
                
                fact_strings = [f['fact'] for f in raw_facts]
                
                # 1. UI Slider = Direct Threshold
                # 0.20 is usually a good "Relevance" cutoff for Cross-Encoders
                ui_threshold = float(self.all_settings.get('safe_sieve_threshold', 0.25))
                
                # 2. Run Sieve
                high_yield_texts = rank_facts_with_cross_encoder(
                    fact_strings, 
                    self.golden_objectives,
                    threshold=ui_threshold
                )
                
                # 3. Reconstruct Objects
                fact_map = {f['fact']: f for f in raw_facts}
                sieved_facts = []
                for text in high_yield_texts:
                    if text in fact_map:
                        sieved_facts.append(fact_map[text])
                        
                self.log(f"   > [Sieve] Retained {len(sieved_facts)}/{len(raw_facts)} Facts.")
            else:
                sieved_facts = raw_facts
        else:
            sieved_facts = raw_facts

        # from utils import save_sieve_report
        # Calculate diff and save
        # report_msg = save_sieve_report(self.deck_name, raw_facts, sieved_facts)
        # self.log(f"   > [System] {report_msg}")
        
        # --- PHASE 3: LIBRARIAN ---
        # Now operates on the smaller, relevant dataset
        self.log(f"--- Phase 3: Librarian ({len(sieved_facts)} facts) ---")
        self._update_global_bar("librarian", 1, 2, "Cleaning Facts...")
        
        clean_facts = self.llm_service.run_librarian(sieved_facts)

        # --- PHASE 4: ASSEMBLY (Grouping) ---
        self.log("--- Phase 4: Assembly ---")
        self._update_global_bar("assembly", 1, 1, "Grouping Facts...")
        
        from prompts import CREATOR_BATCH_PROMPT, CREATOR_SINGLE_VIGNETTE_PROMPT
        
        if clean_facts:
            max_f = int(self.all_settings.get('max_facts_input', 5))
            min_f = int(self.all_settings.get('min_facts_input', 2))
            
            self.log(f"   > [Assembly] Grouping {len(clean_facts)} facts (Target Max: {max_f}).")
            
            # Group the cleaned facts
            groups = self.llm_service.group_facts_by_topic(clean_facts)
            
            # Select Prompt Type based on Batch Size setting
            if int(self.all_settings.get("creator_batch_size", 1)) > 1:
                self.llm_service.prompt_creator = CREATOR_BATCH_PROMPT
            else:
                self.llm_service.prompt_creator = CREATOR_SINGLE_VIGNETTE_PROMPT
        else:
            self.log("❌ [Assembly] No facts remained after Sieve/Librarian.")
            groups = []

        if not groups:
            return None

        # --- PHASE 5: CREATOR ---
        self.log(f"--- Phase 5: Creator ({len(groups)} bundles) ---")
        def creator_cb(curr, total):
            self._update_global_bar("creator", curr, total, f"Generating Card Batch {curr}/{total}")

        raw_cards = []
        gen_mode = self.all_settings.get("card_generation_mode", "")
        
        if "Cloze" in gen_mode:
             if "Paragraph" in gen_mode:
                 raw_cards = self.llm_service.generate_cloze_paragraph_batch(groups, progress_callback=creator_cb)
             else:
                 raw_cards = self.llm_service.generate_cloze_batch(groups)
        else:
            # Uses the batched generator we fixed previously
            raw_cards = self.llm_service.generate_batch(
                groups, 
                self.llm_service.prompt_creator, 
                progress_callback=creator_cb
            )

        # --- PHASE 6: SURGEON & EXPORT ---
        self.log(f"--- Phase 6: Surgeon & Export ---")
        self._update_global_bar("surgeon", 1, 2, "Refining Cards...")

        # Initialize Surgeon with Hybrid Cost Logic
        surgeon = CardSurgeon(self.all_settings)
        final_cards = []
        
        # User's Max Budget for logging
        max_budget = float(self.all_settings.get('max_facts_input', 5))

        for card_wrapper in raw_cards:
             if not card_wrapper or 'args' not in card_wrapper: continue
             
             if card_wrapper.get('name') == 'create_anki_card':
                 args = card_wrapper['args']
                 back_text = args.get('Back', '')
                 
                 # 1. RUN SURGEON (Cost Analysis)
                 bins = surgeon.operate(back_text)
                 
                 if len(bins) > 1:
                     self.log(f"   > [Surgeon] Split triggered (Cost exceeded budget {max_budget})")
                     
                     base_front = args.get('Front', '')
                     topic = args.get('Topic', 'Medical Concept')
                     
                     for i, bin_parts in enumerate(bins):
                         new_args = args.copy()
                         new_back = "\n".join(bin_parts)
                         new_args['Back'] = new_back
                         
                         refined_front = self.llm_service.run_critic(topic, base_front, new_back)
                         
                         new_args['Front'] = f"{refined_front} ({i+1}/{len(bins)})"
                         final_cards.append({"name": card_wrapper['name'], "args": new_args})
                 else:
                     final_cards.append(card_wrapper)
             else:
                 # Pass Cloze/Mermaid cards through untouched
                 final_cards.append(card_wrapper)

        if not final_cards:
            self.log("❌ No valid cards generated.")
            return None

        self._add_notes_to_anki(final_cards)
        self.log(f"SUCCESS: Deck updated in Anki.")
        self._update_global_bar("complete", 1, 1, "Done!")
        
        return "Success"

def generate_all_decks(max_decks, *args):
    # Unpack fixed inputs
    master_files = args[0]
    # args[1] is generate_button (unused here)
    log_out = args[2]
    clip_model = args[3]
    
    # Calculate slices
    # 4 fixed inputs + (max_decks * 2) deck inputs + 1 excluded_indices + settings
    start_decks = 4
    end_decks = 4 + (max_decks * 2)
    deck_data = args[start_decks:end_decks]
    
    excluded_indices = args[end_decks]
    
    # Settings start after excluded_indices
    settings_vals = args[end_decks + 1:]
    
    # --- CRITICAL FIX: keys must match ui.py component order EXACTLY ---
    keys = [
        # AI
        "ai_provider", "ai_model", "ai_api_key",
        # Pipeline
        "auto_curation", "harvester_batch_size", "safe_sieve_threshold", 
        "card_generation_mode", "cloze_syntax_mode", "cloze_keyword_count", "cloze_group_size",# <--- Added these in correct spot
        "creator_batch_size", "min_facts_input", "max_facts_input", "favor_vignettes", "enable_vision_extraction",
        # Surgeon
        "surgeon_word_limit", "surgeon_bullet_cost", "surgeon_sub_bullet_cost",
        # Prompts
        "prompt_harvester", "prompt_librarian", "prompt_creator", "prompt_critic",
        # Styling (Lists)
        "style_anatomy", "style_drugs", "style_pathology",
        # Styling (Colors)
        "c_structure", "c_topic", "c_data", "c_anatomy", "c_pharma", "c_process", "c_pos", "c_neg",
        # General
        "custom_tags", "pdf_language", 
        "content_strategy", "objectives_text_manual"
    ]
    
    # Create the dictionary
    # We use a safe zip here to prevent crashing if lengths mismatch, but they should match now.
    current_settings = dict(zip(keys, settings_vals))
    
    def logger(msg): print(msg); return msg

    configure_tesseract()
    
    # Identify active decks
    deck_configs = []
    for i in range(0, len(deck_data), 2):
        d_title = deck_data[i]
        d_files = deck_data[i+1]
        if d_title and d_files:
            deck_configs.append((d_title, d_files))
    
    # Run processing
    for deck_name, files in deck_configs:
        proc = DeckProcessor(
            deck_name, 
            files, 
            logger, 
            gr.Progress(), 
            current_settings, # Passed as 'all_settings'
            (Path(".pdf"), Path(".ai")), 
            clip_model, 
            excluded_indices
        )
        proc.run()
        
    return "Done", None, gr.update()