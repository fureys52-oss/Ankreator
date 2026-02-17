import gradio as gr
import json
import os
import io
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from datetime import datetime
import functools

# --- IMPORTS ---
from processing import generate_all_decks
from utils import get_system_specs, recommend_local_model, clean_filename_heuristics, update_decks_from_files, load_settings, save_settings, clear_cache
from prompts import (
    HARVESTER_SYSTEM_PROMPT, LIBRARIAN_SYSTEM_PROMPT, 
    CREATOR_SINGLE_VIGNETTE_PROMPT, CRITIC_PROMPT_TEMPLATE
)

# [Top of ui.py]

# --- DEFAULTS CONSTANTS ---
DEFAULTS = {
    # AI
    "ai_provider": "Ollama (Local)",
    "ai_model": "ministral-3:8b",
    "ai_api_key": "",
    
    # Pipeline
    "auto_curation": False,
    "harvester_batch_size": 3,
    "safe_sieve_threshold": 0.25,
    "card_generation_mode": "Basic (QA Bullets)",
    "cloze_syntax_mode": "Combined (All c1)",
    "cloze_keyword_count": 5,
    "cloze_group_size": 2,
    "creator_batch_size": 1,
    "min_facts_input": 1,
    "max_facts_input": 3,
    "enable_vision_extraction": True,
    "favor_vignettes": False,

    # Surgeon
    "surgeon_word_limit": 20,
    "surgeon_bullet_cost": 1.0,
    "surgeon_sub_bullet_cost": 0.5,

    # Styling
    "style_anatomy": "", "style_drugs": "", "style_pathology": "",
    "c_structure": "#93C5FD", "c_topic": "#FDBA74", "c_data": "#2DD4BF", 
    "c_anatomy": "#60A5FA", "c_pharma": "#F472B6", "c_process": "#A5B4FC", 
    "c_pos": "#86EFAC", "c_neg": "#F87171",
    "custom_tags": "", "pdf_language": "English",

    # Prompts
    "prompt_harvester": HARVESTER_SYSTEM_PROMPT,
    "prompt_librarian": LIBRARIAN_SYSTEM_PROMPT,
    "prompt_creator": CREATOR_SINGLE_VIGNETTE_PROMPT,
    "prompt_critic": CRITIC_PROMPT_TEMPLATE
}

# --- HELPER FUNCTIONS ---
def render_file_thumbnails(files):
    if not files: return [], []
    thumbnails = []
    for file_obj in files:
        try:
            doc = fitz.open(file_obj.name)
            for i, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(0.4, 0.4))
                img_data = pix.tobytes("jpg", jpg_quality=80)
                img = Image.open(io.BytesIO(img_data))
                label = f"{Path(file_obj.name).name} - Pg {i+1}"
                thumbnails.append((img, label))
        except Exception as e:
            print(f"Error rendering: {e}")
    return thumbnails, []

def toggle_page_exclusion(evt: gr.SelectData, current_excluded):
    idx = evt.index
    if idx in current_excluded:
        current_excluded.remove(idx)
        action = "Restored"
    else:
        current_excluded.append(idx)
        action = "Removed"
    current_excluded.sort()
    return current_excluded, f"Last Action: {action} Page {idx+1}. \nTotal Excluded: {len(current_excluded)} pages."

# --- CSS ---
CUSTOM_CSS = """
.toast-wrap {
    left: 50% !important;
    top: 50% !important;
    transform: translate(-50%, -50%) !important;
    width: auto !important;
    max-width: 80vw !important;
    z-index: 10000 !important;
}
.toast-body {
    font-size: 1.1rem !important;
    padding: 15px !important;
    border-radius: 8px !important;
}
#gen-btn {
    height: 80px !important; 
    font-size: 1.2rem !important;
}
/* GALLERY STYLES */
#gallery button.excluded-item { 
    filter: grayscale(100%) opacity(0.5) !important; 
    border: 5px solid #ff0000 !important;
    transform: scale(0.95);
    transition: all 0.1s;
}
#gallery button.excluded-item::after { 
    content: "REMOVED"; 
    position: absolute; top: 50%; left: 50%; 
    transform: translate(-50%, -50%); 
    color: white; font-weight: 900; font-size: 1.5em;
    background: rgba(255, 0, 0, 0.8); 
    padding: 10px 20px; border-radius: 8px;
    pointer-events: none; z-index: 999;
}
.model-status {
    font-size: 0.9em;
    color: #666;
    margin-bottom: 10px;
}
"""

def build_ui(settings, version="5.0", **kwargs):
    loaded_settings = load_settings()
    forest_green = gr.themes.Color(
        c50="#f0fdf4",  # Very faint green
        c100="#dcfce7",
        c200="#bbf7d0",
        c300="#86efac",
        c400="#4ade80",
        c500="#1f641f", # <--- Standard Forest Green (Buttons/Headers)
        c600="#175f31",
        c700="#0c4722",
        c800="#134627",
        c900="#0b3a1e", # <--- Deep Forest (Text/Dark Boxes)
        c950="#052e16", # Darkest
    )
    
    # 2. Apply it
    theme = gr.themes.Soft(primary_hue=forest_green)
    # Define visible deck slots (Limit to 20 to keep UI snappy, backend supports 100)
    VISIBLE_DECK_SLOTS = 100
    MAX_DECKS_BACKEND = kwargs.get('max_decks', 100)

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title=f"Ankreator") as app:
        clip_model_status = gr.Markdown("⏳ Loading Local AI models (CLIP + MPNet)...", elem_classes="model-status", visible=False)
        # --- STATE ---
        clip_model_state = gr.State(kwargs.get('clip_model', None)) 
        excluded_indices_state = gr.State([])

        # --- ROW 1: HEADER & STATUS ---
        update_notifier = gr.Markdown(visible=False) # Hidden unless update found
        
        with gr.Row():
            with gr.Column(scale=7):
                # Larger Title
                gr.Markdown(f"<h1 style='font-size: 2rem; margin-bottom: 10px;'>Ankreator</h1>")
                
            
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    **Quick Start:**
                    1. **Create Deck:** Upload PDF, exclude pages visually.
                    2. **Settings:** Tweak AI or Card limits (Auto-Saved).
                    3. **Generate:** Press the big button below.
                    """
                )

        # --- ROW 2: CONTROLS ---
        with gr.Row():
            with gr.Column(scale=5):
                global_status = gr.Textbox(
                     
                    value="Ready", 
                    interactive=False, 
                    show_label=False,
                )
            with gr.Column(scale=5):
                btn_generate = gr.Button("Generate All Decks", variant="primary", scale=1, elem_id="gen-btn")

        # --- ROW 3: MAIN TABS ---
        with gr.Tabs():
            
            # TAB 1: DECK CREATION
            with gr.TabItem("Deck Creation"):
                
                with gr.Row():
                    # LEFT: UPLOAD & EXCLUSION
                    with gr.Column(scale=6, variant="panel"):
                        gr.Markdown("### 1. Upload PDFs")
                        file_input = gr.File(
                            label="Upload PDF(s)", 
                            file_count="multiple", 
                            file_types=[".pdf"],
                            type="filepath",
                            height=200
                        )
                        
                        # --- VISUAL GALLERY (Accordion) ---
                        with gr.Accordion("Exclude Pages (Visual)", open=False):
                            gr.Markdown("Click thumbnails to remove pages from processing.")
                            load_previews_btn = gr.Button("Load Page Previews", variant="secondary")
                            exclusion_status = gr.Markdown("No pages excluded.")
                            page_gallery = gr.Gallery(
                                label="Page Previews", show_label=False, elem_id="gallery", 
                                columns=[3], rows=[2], object_fit="contain", height="auto", allow_preview=False
                            )

                    # RIGHT: STRATEGY & DECK LIST
                    with gr.Column(scale=4, variant="panel"):
                        gr.Markdown("### 2. Deck Configuration")
                        content_strategy = gr.Radio(
                            choices=["Extract All Facts", "Auto-Extract Objectives", "Focus on Provided Objectives"],
                            label="Content Strategy",
                            value=loaded_settings.get("content_strategy", "Extract All Facts"),
                            info="Choose how the system handles learning objectives during extraction."
                        )
                        
                        objectives_manual_input = gr.Textbox(
                            label="Paste Objectives Here",
                            placeholder="- Define the stages of mitosis...\n- Explain the Krebs cycle...",
                            lines=4,
                            visible=(loaded_settings.get("content_strategy") == "Focus on Provided Objectives"),
                            info="Enter specific learning objectives to guide fact extraction. One per line."
                        )
                        
                        def toggle_objectives(choice):
                            return gr.update(visible=(choice == "Focus on Provided Objectives"))
                        with gr.Row():
                                pdf_language = gr.Dropdown([
                                        "English", "Spanish", "French", "German", "Italian", 
                                        "Portuguese", "Chinese (Mandarin)", "Japanese", "Korean", 
                                        "Russian", "Arabic", "Hindi", "Turkish", "Dutch", "Polish",
                                        "Vietnamese", "Indonesian"], label="Language", value=loaded_settings.get("pdf_language", "English"), interactive=True, info="Select the language output for the generated cards.")
                        
                        content_strategy.change(fn=toggle_objectives, inputs=content_strategy, outputs=objectives_manual_input)
                        gr.Markdown("### 3. Name Your Decks")
                        # --- MULTI-DECK ACCORDIONS ---
                        deck_ui_components_for_update = [] # [Acc, Title, File, Acc, Title, File...] (For utils.py)
                        deck_input_components_for_gen = [] # [Title, File, Title, File...] (For processing.py)
                        
                        for i in range(VISIBLE_DECK_SLOTS):
                            with gr.Group(visible=False) as d_group:
                                # [CHANGE] Header using Forest Green (#228b22)
                                d_header = gr.HTML(
                                    f"<div style='background-color: #228b22; color: white; padding: 4px; text-align: center; border-radius: 5px; font-weight: bold;'>Deck {i+1}</div>"
                                )
                                
                                with gr.Row(equal_height=True):
                                    # [CHANGE] Label Box using Deep Forest (#14532d)
                                    gr.HTML(
                                        """<div style='
                                            background-color: #14532d; 
                                            color: white; 
                                            border-radius: 8px; 
                                            height: 100%; 
                                            min-height: 42px; 
                                            display: flex; 
                                            align-items: center; 
                                            justify-content: center; 
                                            font-weight: bold;
                                            font-size: 0.9em;
                                        '>Deck Name</div>"""
                                    )
                                    
                                    d_title = gr.Textbox(
                                        show_label=False, 
                                        interactive=True, 
                                        scale=5
                                    )
                                
                                d_file = gr.File(label="Source File", interactive=False, visible=False)
                                
                                deck_ui_components_for_update.extend([d_group, d_header, d_title, d_file])
                                
                                deck_input_components_for_gen.append(d_title)

                        
                        
                        
            # TAB 2: SETTINGS
            with gr.TabItem("⚙️ Settings"):
                
                # --- AUTO-SAVE STATUS ---
                with gr.Row():
                    save_status_textbox = gr.Textbox(label="Auto-Save Status", value="Ready", interactive=False, scale=4)
                    force_save_btn = gr.Button("Manual Save - Settings are already automatically saved.", variant="secondary", scale=1)

                with gr.Tabs():
                    
                    # A. AI
                    # A. AI
                    # A. AI TAB
                    # A. AI TAB
                    with gr.TabItem("AI & API"):
                        with gr.Row():
                            gr.Markdown("### AI Configuration")
                            btn_reset_ai = gr.Button("↺ Reset to Default", size="sm", variant="secondary")

                        specs = get_system_specs()
                        rec_model_default = recommend_local_model(specs)
                        
                        ai_provider = gr.Dropdown(
                            ["Ollama (Local)", "OpenAI", "Google Gemini", "Anthropic", "Groq", "OpenRouter"], 
                            label="AI Provider", 
                            value=loaded_settings.get("ai_provider", DEFAULTS["ai_provider"])
                        )

                        # --- OLLAMA GROUP ---
                        with gr.Group(visible=(ai_provider.value == "Ollama (Local)")) as ollama_group:
                            # 1. Ollama Instructions
                            gr.HTML("""
                                <div style="background-color: #1f2937; border: 2px solid #22c55e; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                                    <h3 style="margin-top: 0; color: #4ade80;">🦙 Ollama Setup Guide</h3>
                                    <ol style="margin-bottom: 0; color: #e5e7eb;">
                                        <li><b><a href="https://ollama.com/download/windows" target="_blank" style="color: #86efac; text-decoration: underline;">Click Here to Download & Install Ollama</a></b></li>
                                        <li>Run the installer and ensure the Ollama app is running in your taskbar.</li>
                                        <li>Select a recommended model below and click "Download".</li>
                                        <li>Select or Type the model name in below where it says "(Model Name (Select or Type))".</li>
                                        <li>You are good to go!</li>
                                    </ol>
                                </div>
                            """)
                            
                            # 2. Recommendation Engine
                            from utils import get_smart_recommendations, pull_ollama_model, get_ollama_models
                            recs = get_smart_recommendations(specs)
                            radio_choices = [f"{r[0]}  |  {r[1]}  |  {r[2]}" for r in recs]
                            
                            ram_disp = specs.get('gpu_vram_gb', 0) if specs.get('has_nvidia') else specs.get('ram_gb', 0)
                            hardware_type = "VRAM" if specs.get('has_nvidia') else "System RAM"
                            
                            gr.Markdown(f"**Recommended Models for your System** ({ram_disp}GB {hardware_type} Detected):")
                            
                            with gr.Row():
                                model_radio = gr.Radio(choices=radio_choices, label="Select a Recommended Model", interactive=True, scale=4)
                                btn_download_model = gr.Button("⬇️ Download Selected Model", variant="primary", scale=1)
                            
                            download_status = gr.Textbox(label="Status", interactive=False, visible=True)
                            
                            def run_download(selection):
                                if not selection: return "Please select a model first."
                                return pull_ollama_model(selection)

                            btn_download_model.click(run_download, inputs=model_radio, outputs=download_status)

                        # --- CLOUD INSTRUCTIONS (Dynamic) ---
                        # This HTML block will be updated by the python function below
                        cloud_instruction_box = gr.HTML(visible=False)

                        # --- MODEL SELECTION ROW ---
                        with gr.Row(equal_height=True):
                            ai_model = gr.Dropdown(
                                label="Model Name (Select or Type)", 
                                value=loaded_settings.get("ai_model", rec_model_default),
                                allow_custom_value=True,
                                scale=4
                            )
                            # REFRESH BUTTON (Visible only for Ollama ideally, but useful generally)
                            btn_refresh_models = gr.Button("🔄 Refresh List", variant="secondary", scale=1)

                        ai_api_key = gr.Textbox(
                            label="API Key", type="password", 
                            value=loaded_settings.get("ai_api_key", DEFAULTS["ai_api_key"]),
                            visible=False
                        )

                        # --- LOGIC CENTER ---
                        def update_provider_ui(provider):
                            # 1. Data Source
                            provider_data = {
                                "Ollama (Local)": ("https://ollama.com/library", "", "llama3"),
                                "OpenAI": ("https://platform.openai.com/docs/models", "https://platform.openai.com/api-keys", "gpt-4o"),
                                "Google Gemini": ("https://ai.google.dev/gemini-api/docs/models/gemini", "https://aistudio.google.com/app/apikey", "gemini-1.5-pro"),
                                "Anthropic": ("https://docs.anthropic.com/en/docs/models-overview", "https://console.anthropic.com/settings/keys", "claude-3-5-sonnet-20240620"),
                                "Groq": ("https://console.groq.com/docs/models", "https://console.groq.com/keys", "llama3-70b-8192"),
                                "OpenRouter": ("https://openrouter.ai/models", "https://openrouter.ai/keys", "openai/gpt-4o")
                            }
                            
                            provider_defaults = {
                                "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                                "Google Gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.0-pro"],
                                "Anthropic": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
                                "Groq": ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
                                "OpenRouter": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5"]
                            }

                            is_ollama = (provider == "Ollama (Local)")
                            model_url, key_url, example_model = provider_data.get(provider, ("", "", ""))
                            
                            # 2. Build Cloud HTML (If not Ollama)
                            cloud_html = ""
                            if not is_ollama:
                                cloud_html = f"""
                                <div style="background-color: #1f2937; border: 2px solid #3b82f6; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                                    <h3 style="margin-top: 0; color: #60a5fa;">☁️ {provider} Setup Guide</h3>
                                    <ol style="margin-bottom: 0; color: #e5e7eb;">
                                        <li><b><a href="{model_url}" target="_blank" style="color: #93c5fd; text-decoration: underline;">Click here to find {provider} Models</a></b></li>
                                        <li>Type the model name exactly (e.g., <code>{example_model}</code>) in the box below.</li>
                                        <li><b><a href="{key_url}" target="_blank" style="color: #93c5fd; text-decoration: underline;">Click here to get your API Key</a></b></li>
                                        <li>Paste the key into the API Key box.</li>
                                    </ol>
                                </div>
                                """

                            # 3. Fetch Choices
                            new_choices = []
                            if is_ollama:
                                try:
                                    from utils import get_ollama_models
                                    new_choices = get_ollama_models()
                                except: pass
                            else:
                                new_choices = provider_defaults.get(provider, [])

                            # 4. Return Updates
                            return (
                                gr.update(visible=is_ollama),           # Ollama Group
                                gr.update(visible=(not is_ollama), value=cloud_html), # Cloud HTML Box
                                gr.update(choices=new_choices, value=new_choices[0] if new_choices else None), # Dropdown
                                gr.update(visible=(not is_ollama))      # API Key Box
                            )

                        # Wire Provider Change
                        ai_provider.change(
                            fn=update_provider_ui, 
                            inputs=[ai_provider], 
                            outputs=[ollama_group, cloud_instruction_box, ai_model, ai_api_key]
                        )
                        
                        # Wire Refresh Button
                        def refresh_model_list(provider):
                            # Just re-run the update logic for the current provider
                            return update_provider_ui(provider)[2] # Return just the dropdown update
                            
                        btn_refresh_models.click(
                            fn=refresh_model_list,
                            inputs=[ai_provider],
                            outputs=[ai_model]
                        )

                        # Reset Logic
                        btn_reset_ai.click(
                            lambda: [DEFAULTS["ai_provider"], DEFAULTS["ai_model"], DEFAULTS["ai_api_key"]],
                            outputs=[ai_provider, ai_model, ai_api_key]
                        )
                    # B. PIPELINE
                    with gr.TabItem("Pipeline"):
                        with gr.Row():
                         gr.Markdown("### Pipeline Settings")
                         btn_reset_pipe = gr.Button("↺ Reset to Default", size="sm", variant="secondary")
                        auto_curation_toggle = gr.Checkbox(
                            label="Enable Auto-Curation (Remove References)", 
                            value=loaded_settings.get("auto_curation", False),
                            info="Automatically excludes pages that look like bibliographies or TOCs."
                        )
                        harvester_batch_size = gr.Slider(
                            1, 100, value=loaded_settings.get("harvester_batch_size", 3), step=1, 
                            label="Harvester Batch Size", info="Pages per AI prompt. Higher = more context but higher RAM usage."
                        )
                        safe_sieve_threshold = gr.Slider(
                            0.0, 1.0, value=loaded_settings.get("safe_sieve_threshold", 0.25), step=0.01, 
                            label="Safe Sieve Threshold", info="Confidence threshold for filtering content against learning objectives. Lower = more content retained. Higher = stricter filtering."
                        )
                        
                        gr.Markdown("### Card Type")
                        card_generation_mode = gr.Radio(
                            ["Basic (QA Bullets)", "Cloze (Paragraph)"],
                            label="Mode",
                            value=loaded_settings.get("card_generation_mode", "Basic (QA Bullets)"),
                            info="Basic: Standard Q&A. Cloze: Sentence(s) with fill-in-the-blanks."
                        )
                        
                        is_cloze = (loaded_settings.get("card_generation_mode") == "Cloze (Paragraph)")
                        with gr.Group(visible=is_cloze) as cloze_settings_group:
                            with gr.Row():
                                saved_syntax = loaded_settings.get("cloze_syntax_mode", "Sequential (c1, c2, c3)")
                                if "Combined" in saved_syntax and "All" not in saved_syntax: saved_syntax = "Combined (All c1)"
                                
                                cloze_syntax_mode = gr.Dropdown(
                                    ["Combined (All c1)", "Sequential (c1, c2, c3)", "Grouped (c1, c1, c2, c2)", "Alternating (c1, c2, c1, c2)"], 
                                    label="Cloze Pattern", value=saved_syntax, info="Determines how cloze deletions are structured within the card."
                                )
                                cloze_keyword_count = gr.Slider(1, 10, value=loaded_settings.get("cloze_keyword_count", 5), step=1, label="Keywords", info="Number of keywords to convert into cloze deletions per card.")
                                cloze_group_size = gr.Slider(1, 10, value=loaded_settings.get("cloze_group_size", 2), step=1, label="Group Size", info="Number of cloze keywords in one group per cloze deletion.")
                        
                        def toggle_cloze(mode): return gr.update(visible=(mode == "Cloze (Paragraph)"))
                        card_generation_mode.change(fn=toggle_cloze, inputs=card_generation_mode, outputs=cloze_settings_group)
                        
                        creator_batch_size = gr.Slider(1, 100, value=loaded_settings.get("creator_batch_size", 1), step=1, label="Cards per Prompt", info="Number of cards to generate per AI prompt. Higher = Riskier/Faster. Lower = Safer/Slower.")
                        min_facts_input = gr.Slider(1, 100, value=loaded_settings.get("min_facts_input", 1), step=1, label="Min Facts", info="Minimum number of facts to include per card.")
                        max_facts_input = gr.Slider(1, 100, value=loaded_settings.get("max_facts_input", 3), step=1, label="Max Facts", info="Maximum number of facts to include per card.")
                        enable_vision_extraction = gr.Checkbox(label="Extract Images", value=loaded_settings.get("enable_vision_extraction", True), info="Attempt to find and attach relevant images from the PDF.")
                        favor_vignettes = gr.Checkbox(label="Favor Vignettes", value=loaded_settings.get("favor_vignettes", False), visible=False)
                        btn_reset_pipe.click(
                        lambda: [
                            DEFAULTS["auto_curation"], DEFAULTS["harvester_batch_size"], DEFAULTS["safe_sieve_threshold"],
                            DEFAULTS["card_generation_mode"], DEFAULTS["cloze_syntax_mode"], DEFAULTS["cloze_keyword_count"],
                            DEFAULTS["cloze_group_size"], DEFAULTS["creator_batch_size"], DEFAULTS["min_facts_input"],
                            DEFAULTS["max_facts_input"], DEFAULTS["enable_vision_extraction"], DEFAULTS["favor_vignettes"]
                        ],
                        outputs=[
                            auto_curation_toggle, harvester_batch_size, safe_sieve_threshold,
                            card_generation_mode, cloze_syntax_mode, cloze_keyword_count,
                            cloze_group_size, creator_batch_size, min_facts_input,
                            max_facts_input, enable_vision_extraction, favor_vignettes
                        ]
                    )
                        
                    # C. STYLING
                    with gr.TabItem("Styling"):
                        with gr.Row():
                         gr.Markdown("### Styling Configuration")
                         btn_reset_style = gr.Button("↺ Reset to Default", size="sm", variant="secondary")
                        gr.Markdown("### Highlight Lists (Comma Separated)")
                        style_anatomy = gr.Textbox(label="Anatomy Terms", value=loaded_settings.get("style_anatomy", ""), placeholder="heart, lung, aorta...")
                        style_drugs = gr.Textbox(label="Drug Names", value=loaded_settings.get("style_drugs", ""), placeholder="aspirin, ibuprofen...")
                        style_pathology = gr.Textbox(label="Pathology Terms", value=loaded_settings.get("style_pathology", ""), placeholder="cancer, stenosis...")
                        
                        gr.Markdown("### Palette")
                        with gr.Row():
                            c_structure = gr.ColorPicker(label="Structure", value=loaded_settings.get("c_structure", "#93C5FD")) 
                            c_topic = gr.ColorPicker(label="Topic", value=loaded_settings.get("c_topic", "#FDBA74"))     
                            c_data = gr.ColorPicker(label="Data", value=loaded_settings.get("c_data", "#2DD4BF"))      
                            c_anatomy = gr.ColorPicker(label="Anatomy", value=loaded_settings.get("c_anatomy", "#60A5FA"))   
                        with gr.Row():
                            c_pharma = gr.ColorPicker(label="Pharma", value=loaded_settings.get("c_pharma", "#F472B6"))    
                            c_process = gr.ColorPicker(label="Process", value=loaded_settings.get("c_process", "#A5B4FC"))   
                            c_pos = gr.ColorPicker(label="Positive", value=loaded_settings.get("c_pos", "#86EFAC"))      
                            c_neg = gr.ColorPicker(label="Negative", value=loaded_settings.get("c_neg", "#F87171"))      
                        
                        custom_tags_textbox = gr.Textbox(label="Anki Tags", value=loaded_settings.get("custom_tags", ""), info="Tags added to every card (comma separated).")
                        btn_reset_style.click(
                        lambda: [
                            DEFAULTS["style_anatomy"], DEFAULTS["style_drugs"], DEFAULTS["style_pathology"],
                            DEFAULTS["c_structure"], DEFAULTS["c_topic"], DEFAULTS["c_data"],
                            DEFAULTS["c_anatomy"], DEFAULTS["c_pharma"], DEFAULTS["c_process"],
                            DEFAULTS["c_pos"], DEFAULTS["c_neg"], DEFAULTS["custom_tags"], DEFAULTS["pdf_language"]
                        ],
                        outputs=[
                            style_anatomy, style_drugs, style_pathology,
                            c_structure, c_topic, c_data, c_anatomy, c_pharma, c_process, c_pos, c_neg,
                            custom_tags_textbox
                        ]
                    )
                    # D. SURGEON
                    with gr.TabItem("Surgeon"):
                        with gr.Row():
                         gr.Markdown("### Surgeon Logic")
                         btn_reset_surg = gr.Button("↺ Reset to Default", size="sm", variant="secondary")
                        gr.Markdown("Controls how large cards are split into smaller ones. This is for fine tuning card granularity. Define here what a single fact unit is.")
                        surgeon_word_limit = gr.Slider(10, 50, value=loaded_settings.get("surgeon_word_limit", 20), step=1, label="Words/Unit", info="Maximum words per fact unit. What is the max number of words that defines a single fact?")
                        surgeon_bullet_cost = gr.Slider(0.1, 5.0, value=loaded_settings.get("surgeon_bullet_cost", 1.0), step=0.1, label="Bullet Cost", info="How many facts are in one bullet point? Low = More bullets, High = Fewer bullets.")
                        surgeon_sub_bullet_cost = gr.Slider(0.1, 5.0, value=loaded_settings.get("surgeon_sub_bullet_cost", 0.5), step=0.1, label="Sub-Bullet Cost", info="How many facts are in one sub-bullet point? Low = More sub-bullets, High = Fewer sub-bullets.")
                        btn_reset_surg.click(
                        lambda: [DEFAULTS["surgeon_word_limit"], DEFAULTS["surgeon_bullet_cost"], DEFAULTS["surgeon_sub_bullet_cost"]],
                        outputs=[surgeon_word_limit, surgeon_bullet_cost, surgeon_sub_bullet_cost]
                    )
                    # E. PROMPTS
                    with gr.TabItem("Prompts"):
                        with gr.Row():
                         gr.Markdown("### System Prompts")
                         btn_reset_prompt = gr.Button("↺ Reset to Default", size="sm", variant="secondary")
                        prompt_harvester = gr.TextArea(label="Harvester", value=loaded_settings.get("prompt_harvester", HARVESTER_SYSTEM_PROMPT), lines=3)
                        prompt_librarian = gr.TextArea(label="Librarian", value=loaded_settings.get("prompt_librarian", LIBRARIAN_SYSTEM_PROMPT), lines=3)
                        prompt_creator = gr.TextArea(label="Creator", value=loaded_settings.get("prompt_creator", CREATOR_SINGLE_VIGNETTE_PROMPT), lines=3)
                        prompt_critic = gr.TextArea(label="Critic", value=loaded_settings.get("prompt_critic", CRITIC_PROMPT_TEMPLATE), lines=3)
                        btn_reset_prompt.click(
                        lambda: [DEFAULTS["prompt_harvester"], DEFAULTS["prompt_librarian"], DEFAULTS["prompt_creator"], DEFAULTS["prompt_critic"]],
                        outputs=[prompt_harvester, prompt_librarian, prompt_creator, prompt_critic]
                    )
                    # F. SESSION LOGS
                    with gr.TabItem("Session Logs"):
                        with gr.Row():
                            btn_clear_cache = gr.Button("Clear Cache", size="sm", variant="secondary")
                            btn_copy_log = gr.Button("Copy Log", size="sm", variant="secondary")
                        
                        cache_status = gr.Textbox(value="Cache: Ready", label="Status", interactive=False, show_label=False)
                        
                        log_display = gr.Code(
                            label="Execution Log", 
                            language="markdown", 
                            interactive=False, 
                            lines=20, 
                            value="System Ready..."
                        )

        # --- EVENT WIRING: Gallery ---
        load_previews_btn.click(fn=render_file_thumbnails, inputs=file_input, outputs=[page_gallery, excluded_indices_state])
        page_gallery.select(fn=toggle_page_exclusion, inputs=[excluded_indices_state], outputs=[excluded_indices_state, exclusion_status])

        # --- EVENT WIRING: Deck Populator ---
        # FIXED: Removed 'file_input' from outputs to prevent validation loops.
        file_input.change(
            fn=functools.partial(update_decks_from_files, max_decks=VISIBLE_DECK_SLOTS),
            inputs=[file_input], 
            outputs=deck_ui_components_for_update # Only update the deck slots
        )

        # --- EVENT WIRING: Auto-Save ---
        SETTINGS_KEYS = [
            "deck_name_prefix",
            "ai_provider", "ai_model", "ai_api_key",
            "auto_curation", "harvester_batch_size", "safe_sieve_threshold", 
            "card_generation_mode", "cloze_syntax_mode", "cloze_keyword_count", "cloze_group_size",
            "creator_batch_size", "min_facts_input", "max_facts_input", "favor_vignettes", "enable_vision_extraction",
            "surgeon_word_limit", "surgeon_bullet_cost", "surgeon_sub_bullet_cost",
            "prompt_harvester", "prompt_librarian", "prompt_creator", "prompt_critic",
            "style_anatomy", "style_drugs", "style_pathology",
            "c_structure", "c_topic", "c_data", "c_anatomy", "c_pharma", "c_process", "c_pos", "c_neg",
            "custom_tags", "pdf_language", 
            "content_strategy", "objectives_text_manual"
        ]

        deck_prefix_state = gr.State(loaded_settings.get("deck_name_prefix", "Medical_Concepts"))

        settings_components = [
            deck_prefix_state, 
            ai_provider, ai_model, ai_api_key,
            auto_curation_toggle, harvester_batch_size, safe_sieve_threshold, 
            card_generation_mode, cloze_syntax_mode, cloze_keyword_count, cloze_group_size,
            creator_batch_size, min_facts_input, max_facts_input, favor_vignettes, enable_vision_extraction,
            surgeon_word_limit, surgeon_bullet_cost, surgeon_sub_bullet_cost,
            prompt_harvester, prompt_librarian, prompt_creator, prompt_critic,
            style_anatomy, style_drugs, style_pathology,
            c_structure, c_topic, c_data, c_anatomy, c_pharma, c_process, c_pos, c_neg,
            custom_tags_textbox, pdf_language,
            content_strategy, objectives_manual_input
        ]

        def save_current_settings(*args):
            try:
                current_values = dict(zip(SETTINGS_KEYS, args))
                save_settings(current_values)
                return f"Saved at {datetime.now().strftime('%H:%M:%S')}"
            except Exception as e:
                return f"Error: {e}"

        for comp in settings_components:
            if isinstance(comp, (gr.State,)): continue
            comp.change(fn=save_current_settings, inputs=settings_components, outputs=save_status_textbox)
        
        force_save_btn.click(fn=save_current_settings, inputs=settings_components, outputs=save_status_textbox)

        # --- EVENT WIRING: Generation ---
        
        processing_settings_components = [
            ai_provider, ai_model, ai_api_key,
            auto_curation_toggle, harvester_batch_size, safe_sieve_threshold, 
            card_generation_mode, cloze_syntax_mode, cloze_keyword_count, cloze_group_size,
            creator_batch_size, min_facts_input, max_facts_input, favor_vignettes, enable_vision_extraction,
            surgeon_word_limit, surgeon_bullet_cost, surgeon_sub_bullet_cost,
            prompt_harvester, prompt_librarian, prompt_creator, prompt_critic,
            style_anatomy, style_drugs, style_pathology,
            c_structure, c_topic, c_data, c_anatomy, c_pharma, c_process, c_pos, c_neg,
            custom_tags_textbox, pdf_language,
            content_strategy, objectives_manual_input
        ]
        
        def run_generation(files, clip_model, excluded_indices, *all_args):
            # 1. SPLIT ARGS
            # We only passed TITLES (d_title) to this function from the UI, so count is just the number of slots.
            num_deck_inputs = VISIBLE_DECK_SLOTS 
            
            # Extract the titles list
            titles = all_args[:num_deck_inputs] 
            
            # The rest are settings
            proc_settings_values = all_args[num_deck_inputs:]
            
            # 2. CONSTRUCT DECK CONFIGS
            deck_args_list = []
            files = files or []
            
            for i in range(MAX_DECKS_BACKEND):
                if i < len(files):
                    # Get the title if available (and if i is within the visible slots range)
                    user_title = ""
                    if i < len(titles):
                        user_title = titles[i]
                    
                    # Fallback to filename if title is empty or missing
                    t = user_title if user_title else Path(files[i]).stem
                    
                    # Map the title to the file path from the main list
                    deck_args_list.extend([t, [files[i]]])
                else:
                    # Empty slot
                    deck_args_list.extend([None, None])

            # 3. Call Generator
            result = generate_all_decks(
                MAX_DECKS_BACKEND,
                files, 
                None, 
                lambda x: x,
                clip_model,
                *deck_args_list, 
                excluded_indices, 
                *proc_settings_values
            )
            
            yield result[0], result[1], "Generation Complete"

        gen_event = btn_generate.click(
            fn=run_generation,
            inputs=[file_input, clip_model_state, excluded_indices_state] + deck_input_components_for_gen + processing_settings_components,
            # Added global_status to outputs
            outputs=[log_display, cache_status, global_status]
        )
        
        btn_copy_log.click(None, [log_display], js="(x) => { navigator.clipboard.writeText(x); alert('Log Copied to Clipboard!'); }")
        btn_clear_cache.click(fn=lambda: clear_cache(*kwargs.get('cache_dirs', ())), outputs=[cache_status])
        
        app.load(kwargs.get('update_checker_func', lambda: gr.update()), None, update_notifier)
        app.load(kwargs.get('load_clip_model_func', lambda: ({'model': None}, "AI Idle")), None, [clip_model_state, clip_model_status])

    return app