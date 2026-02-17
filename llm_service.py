import os
import json
import re
import ast
import numpy as np
import time
import gradio as gr
from litellm import completion
import requests
from prompts import (
    HARVESTER_SYSTEM_PROMPT, 
    LIBRARIAN_SYSTEM_PROMPT, 
    CREATOR_SINGLE_VIGNETTE_PROMPT,
    CREATOR_BATCH_PROMPT, # New
    CLOZE_SINGLE_PROMPT,
    CRITIC_PROMPT_TEMPLATE,
    OBJECTIVE_FINDER_PROMPT,
    SCOUT_STRICT_SCHEMA,
    SINGLE_CARD_SCHEMA,
    MULTI_CARD_SCHEMA, # New
    SINGLE_CLOZE_SCHEMA,
    CRITIC_SCHEMA,
    LIBRARIAN_STRICT_SCHEMA,
    OBJECTIVE_SCHEMA,
    CREATOR_CLOZE_PARAGRAPH_PROMPT,
    CLOZE_PARAGRAPH_SCHEMA,
    MAPPER_JUDGE_PROMPT
)

class LLMService:
    def __init__(self, logger_func, settings_dict=None):
        self.log = logger_func
        self.settings = settings_dict or {}
        
        # 1. AI Provider Setup
        self.provider = self.settings.get("ai_provider", "Ollama (Local)")
        self.model_name = self.settings.get("ai_model", "ministral-3:8b-instruct-2512-q4_K_M").strip()
        self.api_key = self.settings.get("ai_api_key", "")
        self.prompt_cloze_paragraph = self.settings.get("prompt_cloze_paragraph", CREATOR_CLOZE_PARAGRAPH_PROMPT)

        self.api_base = None
        if self.provider == "Ollama (Local)":
            if not self.model_name.startswith("ollama/"):
                self.model_name = f"ollama/{self.model_name}"
            self.api_base = "http://localhost:11434"
        elif self.provider == "OpenAI":
            os.environ["OPENAI_API_KEY"] = self.api_key
        elif self.provider == "Google Gemini":
            os.environ["GEMINI_API_KEY"] = self.api_key
            if not self.model_name.startswith("gemini/"):
                self.model_name = f"gemini/{self.model_name}"
        elif self.provider == "Anthropic":
             os.environ["ANTHROPIC_API_KEY"] = self.api_key
        elif self.provider == "Groq":
             os.environ["GROQ_API_KEY"] = self.api_key
             if not self.model_name.startswith("groq/"):
                 self.model_name = f"groq/{self.model_name}"
        elif self.provider == "OpenRouter":
             os.environ["OPENROUTER_API_KEY"] = self.api_key
             if not self.model_name.startswith("openrouter/"):
                 self.model_name = f"openrouter/{self.model_name}"

        # 2. Load Prompt Overrides (or defaults)
        self.prompt_harvester = HARVESTER_SYSTEM_PROMPT
        self.prompt_librarian = self.settings.get("prompt_librarian", LIBRARIAN_SYSTEM_PROMPT)
        self.prompt_creator = self.settings.get("prompt_creator", CREATOR_SINGLE_VIGNETTE_PROMPT)
        self.prompt_critic = self.settings.get("prompt_critic", CRITIC_PROMPT_TEMPLATE)

        # 3. Logic Configuration
        self.harvester_batch_size = int(self.settings.get("harvester_batch_size", 3))
        self.creator_batch_size = int(self.settings.get("creator_batch_size", 1))
        self.min_facts = int(self.settings.get("min_facts_input", 2))
        self.max_facts = int(self.settings.get("max_facts_input", 5))
        self.sieve_threshold = float(self.settings.get("safe_sieve_threshold", 0.25))

    def _handle_api_error(self, e, provider, model):
        """Generates user-friendly popups for common errors."""
        e_str = str(e).lower()
        
        # 1. MODEL NOT FOUND
        if "not found" in e_str or "404" in e_str or "does not exist" in e_str:
            links = {
                "openai": "https://platform.openai.com/docs/models",
                "anthropic": "https://docs.anthropic.com/en/docs/models-overview",
                "gemini": "https://ai.google.dev/gemini-api/docs/models/gemini",
                "groq": "https://console.groq.com/docs/models",
                "openrouter": "https://openrouter.ai/models",
                "ollama": "https://ollama.com/library"
            }
            link = links.get(provider, "https://google.com/search?q=" + provider + "+models")
            
            msg = f"🛑 Model '{model}' not found for {provider}.\n"
            if provider == "ollama":
                msg += f"Try running: 'ollama pull {model}' in your terminal.\n"
            msg += f"Verify model name here: {link}"
            
            raise gr.Error(msg)

        # 2. AUTHENTICATION ERRORS
        if "401" in e_str or "auth" in e_str or "api key" in e_str:
            raise gr.Error(f"🛑 Authentication Failed for {provider}.\nCheck your API Key settings.", duration=None)

        # 3. OLLAMA CONNECTION
        if "connection refused" in e_str or "cannot connect" in e_str:
            raise gr.Error("🛑 Cannot connect to Ollama.\nIs the Ollama app running on your computer?", duration=None)

        # 4. CONTEXT WINDOW (Token Limit)
        if "context_length_exceeded" in e_str or "context window" in e_str or "too many tokens" in e_str:
            raise gr.Error(f"🛑 Context Limit Exceeded ({model}).\nPlease reduce 'Harvester Batch Size' in the Pipeline settings.", duration=None)

        # 5. GENERIC FALLBACK
        self.log(f"   > [API Error] {e}")
        return None

    def _call_llm(self, messages, temperature=0.7, schema=None, json_mode=False):
        # --- FIX: Auto-enable JSON mode if schema is present ---
        if schema is not None:
            json_mode = True

        PROVIDER_MAP = {
            "Ollama (Local)": "ollama", "OpenAI": "openai", "Google Gemini": "gemini",
            "Anthropic": "anthropic", "Groq": "groq", "OpenRouter": "openrouter"
        }
        litellm_provider = PROVIDER_MAP.get(self.provider, "ollama")
        
        # --- GUARD RAIL 1: Empty Key Check ---
        if litellm_provider != "ollama":
            if not self.api_key or "sk-" in self.api_key[:3] and len(self.api_key) < 10:
                raise gr.Error(f"🛑 Invalid API Key for {self.provider}.\nPlease enter a valid key in the Settings.", duration=None)

        # Clean Model Name
        final_model = self.model_name
        if "/" in final_model and litellm_provider in final_model.lower():
            _, final_model = final_model.split("/", 1)

        # Traffic Control
        if litellm_provider != "ollama": time.sleep(1.2)
        valid_api_key = self.api_key
        if litellm_provider == "ollama":
            valid_api_key = None 
        
        # Prepare arguments
        kwargs = {
            "model": final_model,
            "messages": messages,
            "temperature": temperature,
            "api_key": valid_api_key,
            "custom_llm_provider": litellm_provider,
        }
        if litellm_provider == "ollama":
            # Pass extra headers/body params depending on how litellm handles it
            # For direct Ollama calls via litellm, we can try adding it to 'extra_body'
            kwargs["extra_body"] = {"keep_alive": "60m"}

        if litellm_provider == "ollama" and json_mode:
            kwargs["format"] = "json"

        if json_mode and schema:
            if litellm_provider in ["openai", "groq", "openrouter", "ollama"]:
                 kwargs["response_format"] = {"type": "json_object"}
            schema_instruction = f"\n\nOutput MUST be a valid JSON object matching this schema:\n{json.dumps(schema, indent=2)}"
            messages[-1]["content"] += schema_instruction

        # --- RETRY LOOP ---
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = completion(**kwargs)
                content = response.choices[0].message.content
                if json_mode: return self._robust_json_parse(content, schema)
                return content

            except Exception as e:
                # RATE LIMIT HANDLING
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if attempt < max_retries:
                        self.log(f"   > [⚠️ Rate Limit] Pausing for 60s (Attempt {attempt}/{max_retries})...")
                        raise gr.warning("⚠️ Rate limit reached. Waiting 60 seconds before retrying...", duration=None)
                        time.sleep(60)
                        continue
                
                # ALL OTHER ERRORS
                self._handle_api_error(e, litellm_provider, final_model)
                return None 
                    
        return None

    def _robust_json_parse(self, text, schema=None):
        # [FIX] Handle case where LLM already returned a Dict
        if isinstance(text, dict):
            return text
            
        if not text: return None
        
        # Now safe to assume it's a string
        text = text.strip()
        
        # 1. Strip Markdown Code Blocks
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match: text = match.group(1).strip()

        # 2. Strategy A: Direct Parse
        try: return json.loads(text)
        except: pass
        
        # 3. Strategy B: "Dirty" Fixes
        try:
            # Fix unescaped newlines inside strings
            clean_text = text.replace('\n', '\\n').replace('\r', '')
            return json.loads(clean_text)
        except: pass

        # 4. Strategy C: Python Eval
        try: return ast.literal_eval(text)
        except: pass

        return None

    def extract_objectives(self, text_chunk):
        prompt = OBJECTIVE_FINDER_PROMPT.replace("{full_text}", text_chunk[:8000])
        response = self._call_llm([{"role": "user", "content": prompt}], temperature=0.1, schema=OBJECTIVE_SCHEMA)
        if response and "objectives" in response:
            return response["objectives"]
        return []

    def run_scout_summary(self, text_chunk):
        """
        Scans the intro text to identify the main topics/chapters.
        """
        prompt = f"""
        Analyze the following text (start of a document) and identify the main subject and 3-5 sub-topics.
        
        TEXT:
        "{text_chunk[:3000]}"...
        
        Return JSON exactly like this:
        {{
            "subject": "Main Subject Name",
            "topics": ["Sub-topic 1", "Sub-topic 2", "Sub-topic 3"]
        }}
        """
        
        # 1. Call LLM with JSON Mode enforced
        response = self._call_llm(
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.3,
            json_mode=True, # <--- Critical: Forces the parser to run
            schema={
                "type": "object", 
                "properties": {
                    "subject": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}}
                }
            }
        )
        
        # 2. Defensive Check: Did we get a String back instead of a Dict?
        if isinstance(response, str):
            # Try to parse it one last time
            parsed = self._robust_json_parse(response)
            if parsed:
                response = parsed
            else:
                self.log(f"   > [Scout Error] Could not parse response: {response[:100]}...")
                return ["General Concepts"] # Fallback to prevent crash

        # 3. Handle None or Missing Keys
        if not response or "topics" not in response:
            return ["General Concepts"]
            
        return response["topics"]

    def run_markdown_harvester(self, markdown_text, progress_callback=None):
        import re
        
        # 1. Split text into logical pages
        raw_pages = re.split(r'(?=\n--- Page \d+ ---)', markdown_text)
        pages = [p for p in raw_pages if p.strip()]

        all_facts = []
        chunks = []
        
        # [FIX] Chunk by PAGE COUNT (Batch Size), not Character Limit
        # This respects the "Harvester Batch Size" slider from UI
        batch_size = int(self.settings.get("harvester_batch_size", 3))
        if batch_size < 1: batch_size = 1

        # Create chunks of N pages
        chunks = ["".join(pages[i:i + batch_size]) for i in range(0, len(pages), batch_size)]

        total_batches = len(chunks)
        self.log(f"   > [Harvester] Processing {len(pages)} pages in {total_batches} batches (Size: {batch_size}).")

        # 2. Process Batches
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, total_batches)

            prompt = self.prompt_harvester.format(text_chunk=chunk)
            
            schema = {
                "type": "object", 
                "properties": {
                    "data": {
                        "type": "array", 
                        "items": {
                            "type": "object", 
                            "properties": {
                                "subject": {"type": "string"}, 
                                "fact": {"type": "string"}, 
                                "page_num": {"type": "integer"}
                            }, 
                            "required": ["subject", "fact", "page_num"]
                        }
                    }
                }, 
                "required": ["data"]
            }

            response = self._call_llm(
                [{"role": "user", "content": prompt}], 
                temperature=0.1, 
                schema=schema
            )
            
            if response and "data" in response:
                batch_facts = response['data']
                self.log(f"   > [Harvester] Batch {i+1}/{total_batches}: Found {len(batch_facts)} facts.")
                all_facts.extend(batch_facts)
            else:
                self.log(f"   > [Harvester] Batch {i+1} yielded no data.")
                
        return all_facts

    # [REPLACEMENT] - Strict Sieve (No "Unmapped" Bucket)
    # [UPDATED] - Accepts min_threshold from UI
    def group_facts_by_objective(self, facts, objectives, min_threshold=0.25):
        """
        Maps facts to objectives using MPNet (High Precision).
        STRICT MODE: Facts below 'min_threshold' are permanently deleted.
        """
        self.log(f"   > [Assembly] Mapping {len(facts)} facts to {len(objectives)} objectives (Min Score: {min_threshold})...")
        
        if not objectives: return self.group_facts_by_topic(facts)

        # [CHANGE 1] STRICT LOAD - No Fallback
        # If this fails, we WANT it to error out so we know MPNet is broken.
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
        
        # Embeddings
        obj_vectors = model.encode(objectives, normalize_embeddings=True)
        
        fact_texts = [f['fact'] for f in facts]
        if not fact_texts: return []
        
        fact_vectors = model.encode(fact_texts, normalize_embeddings=True)

        # [CHANGE 2] DIAGNOSTICS - Why is the score zero?
        # This will print the first comparison to your logs.
        if len(obj_vectors) > 0 and len(fact_vectors) > 0:
            test_score = np.dot(obj_vectors[0], fact_vectors[0])
            self.log(f"   > [DEBUG DIAGNOSTIC] -----------------------------")
            self.log(f"   > [DEBUG] Objective [0]: '{objectives[0]}'")
            self.log(f"   > [DEBUG] Fact [0]: '{fact_texts[0]}'")
            self.log(f"   > [DEBUG] Dot Product Score: {test_score}")
            self.log(f"   > [DEBUG DIAGNOSTIC] -----------------------------")

        buckets = {obj: [] for obj in objectives}
        
        # Thresholds
        MIN_THRESHOLD = min_threshold
        MAX_THRESHOLD = 0.50 # We keep the auto-accept high to be safe
        
        tossed_count = 0
        mapped_count = 0
        judge_calls = 0
        
        for i, fact in enumerate(facts):
            scores = np.dot(obj_vectors, fact_vectors[i])
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            
            # [USE DYNAMIC THRESHOLD]
            if best_score < MIN_THRESHOLD:
                tossed_count += 1
                continue
                
            elif best_score > MAX_THRESHOLD:
                target_obj = objectives[best_idx]
                buckets[target_obj].append(fact)
                mapped_count += 1
                
            else:
    # CALL JUDGE
                top_3_indices = np.argsort(scores)[-3:][::-1]
                
                # Create indexed options for the LLM
                options_text = ""
                for local_id, true_idx in enumerate(top_3_indices):
                    options_text += f"Option {local_id}: {objectives[true_idx]}\n"

                prompt = MAPPER_JUDGE_PROMPT.format(fact=fact['fact'], options_list=options_text)
                
                # Update schema to expect an integer
                schema = {"type": "object", "properties": {"best_match_id": {"type": "integer"}}, "required": ["best_match_id"]}
                
                decision = self._call_llm([{"role": "user", "content": prompt}], temperature=0.0, schema=schema)
                
                selection = decision.get("best_match_id", -1)
                
                # Validate the integer
                if isinstance(selection, int) and 0 <= selection < len(top_3_indices):
                    chosen_real_index = top_3_indices[selection]
                    target_obj = objectives[chosen_real_index]
                    buckets[target_obj].append(fact)
                    mapped_count += 1
                else:
                    tossed_count += 1

        self.log(f"   > [Assembly] Stats: {mapped_count} Mapped | {tossed_count} Tossed (< {MIN_THRESHOLD}) | {judge_calls} Judge Calls.")

        # Convert Buckets to Bundles
        bundles = []
        for obj, assigned_facts in buckets.items():
            if not assigned_facts: continue
            
            chunk_size = self.max_facts
            
            for i in range(0, len(assigned_facts), chunk_size):
                chunk = assigned_facts[i : i + chunk_size]
                pg_nums = sorted(list(set(f['page_num'] for f in chunk)))
                
                bundles.append({
                    "subject": obj,
                    "context": "Learning Objective",
                    "group_id": f"OBJ_{len(bundles)}",
                    "facts": chunk,
                    "page_nums": pg_nums
                })
        
        return bundles 
    # [Targeted Change: Batched Librarian]
    def run_librarian(self, raw_facts):
        # 1. Deduplicate by exact string match first
        unique_facts = {}
        for f in raw_facts:
            unique_facts[f['fact']] = f

        cleaned_facts = list(unique_facts.values())
        self.log(f"   > [Librarian] Deduplicated initial list: {len(raw_facts)} -> {len(cleaned_facts)}")

        # 2. Process in Batches
        BATCH_SIZE = 15
        final_facts = []
        
        # Helper to chunk the list
        chunks = [cleaned_facts[i:i + BATCH_SIZE] for i in range(0, len(cleaned_facts), BATCH_SIZE)]
        
        self.log(f"   > [Librarian] Processing {len(cleaned_facts)} facts in {len(chunks)} batches...")

        for i, chunk in enumerate(chunks):
            # [FIX 1] Input Formatting: Pass actual JSON so the LLM can index them easily
            # We map just the text to keep tokens low
            chunk_texts = [f['fact'] for f in chunk]
            input_json = json.dumps(chunk_texts, indent=2)
            
            # Use the Prompt
            prompt = self.prompt_librarian.replace("{facts_json}", input_json)
            
            # [FIX 2] Schema: Use the STRICT schema that expects Tags (KEEP/NOISE)
            # This matches LIBRARIAN_STRICT_SCHEMA from prompts.py
            schema = {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tag": {"type": "string", "enum": ["KEEP", "NOISE"]}
                            },
                            "required": ["tag"]
                        }
                    }
                },
                "required": ["data"]
            }

            try:
                # Call LLM
                response = self._call_llm(
                    [{"role": "user", "content": prompt}], 
                    temperature=0.0,  # Strict temperature for classification
                    schema=schema
                )
                
                # [FIX 3] Zip & Filter Logic
                # We align the LLM's tags with our original chunk
                if response and "data" in response:
                    tags = response['data']
                    
                    # If model returned fewer tags than facts, pad with "KEEP" (Safety)
                    while len(tags) < len(chunk):
                        tags.append({"tag": "KEEP"})
                        
                    for original_fact, tag_obj in zip(chunk, tags):
                        # [FIX] Handle empty strings or missing keys
                        decision = tag_obj.get('tag', 'KEEP')
                        if not decision or not isinstance(decision, str):
                            decision = 'KEEP'
                            
                        if decision.upper() == 'KEEP':
                            final_facts.append(original_fact)
                else:
                    # Fallback: Model failed to produce JSON data, keep everything
                    final_facts.extend(chunk)

            except Exception as e:
                self.log(f"   > [Librarian] Error in batch {i+1}: {e}. Keeping all facts.")
                final_facts.extend(chunk)

        return final_facts

    def group_facts_by_topic(self, tagged_facts):
        def slugify(text):
            clean = re.sub(r'[^\w\s-]', '', text).strip().lower()
            return re.sub(r'[-\s]+', '_', clean)

        self.log(f"   > [Assembly] Starting Assembly with {len(tagged_facts)} facts.")
        
        indexed_facts = []
        for f in tagged_facts:
            if len(f.get('fact', '')) < 10: continue 
            sub = f.get('subject', 'General')
            if sub in ["None", "Unknown"] or len(sub) < 3: sub = "General"
            f['subject'] = sub 
            indexed_facts.append(f)

        micro_bundles = {}
        for f in indexed_facts:
            key = (f.get('page_num', 0), f['subject'])
            if key not in micro_bundles: micro_bundles[key] = []
            micro_bundles[key].append(f)

        bundles = []
        # Chunk using Settings (Max Facts)
        for (pg, subj), facts in micro_bundles.items():
            chunks = [facts[i:i+self.max_facts] for i in range(0, len(facts), self.max_facts)]
            for chunk in chunks:
                bundles.append({
                    "subject": subj, "Subject": subj,
                    "page_nums": [pg], "Page_Nums": [pg],
                    "facts": chunk, "Facts": chunk,
                    "centroid": None
                })

        # Vertical Merge
        bundles.sort(key=lambda x: (x['subject'], x['page_nums'][0]))
        merged_v = []
        if bundles:
            curr = bundles[0]
            for next_b in bundles[1:]:
                is_adj = abs(curr['page_nums'][-1] - next_b['page_nums'][0]) <= 1
                # Merge if sum is less than Max Settings
                if (curr['subject'] == next_b['subject'] and is_adj and len(curr['facts']) + len(next_b['facts']) <= self.max_facts):
                    curr['facts'] += next_b['facts']
                    curr['page_nums'] = sorted(list(set(curr['page_nums'] + next_b['page_nums'])))
                    continue
                merged_v.append(curr)
                curr = next_b
            merged_v.append(curr)
        bundles = merged_v

        # Horizontal Merge
        bundles.sort(key=lambda x: x['page_nums'][0])
        merged_h = []
        if bundles:
            curr = bundles[0]
            for next_b in bundles[1:]:
                is_same_page = curr['page_nums'][0] == next_b['page_nums'][0]
                # Merge if sum is less than Max Settings AND current is less than Min Settings
                if (is_same_page and len(curr['facts']) < self.min_facts and len(curr['facts']) + len(next_b['facts']) <= self.max_facts):
                    curr['facts'] += next_b['facts']
                    if curr['subject'] != next_b['subject']:
                         curr['subject'] = f"{curr['subject']} & {next_b['subject']}"
                    continue
                merged_h.append(curr)
                curr = next_b
            merged_h.append(curr)
        bundles = merged_h

        # Semantic Merge
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
            for b in bundles:
                texts = [f['fact'] for f in b['facts']]
                if texts: 
                    mean_vec = np.mean(model.encode(texts), axis=0)
                    norm = np.linalg.norm(mean_vec)
                    b['centroid'] = mean_vec / norm if norm > 0 else mean_vec

            final_pass = []
            while bundles:
                curr = bundles.pop(0)
                if len(curr['facts']) >= self.min_facts:
                    final_pass.append(curr)
                    continue
                
                best_idx, best_score = -1, -1
                for i, candidate in enumerate(bundles):
                    # Merge if sum is less than Max Settings
                    if len(curr['facts']) + len(candidate['facts']) > self.max_facts: continue
                    if curr.get('centroid') is None or candidate.get('centroid') is None: continue
                    
                    score = np.dot(curr['centroid'], candidate['centroid'])
                    if score > best_score:
                        best_score = score
                        best_idx = i
                
                if best_idx != -1 and best_score > 0.45:
                    target = bundles[best_idx]
                    target['facts'] += curr['facts']
                    target['page_nums'] = sorted(list(set(target['page_nums'] + curr['page_nums'])))
                    new_mean = (target['centroid'] + curr['centroid']) / 2
                    norm = np.linalg.norm(new_mean)
                    target['centroid'] = new_mean / norm if norm > 0 else new_mean
                else:
                    final_pass.append(curr)
            bundles = final_pass
        except Exception as e:
            self.log(f"   > [Assembly] Vector merge failed ({e}), using simple grouping.")

        for b in bundles:
            b['group_id'] = f"{slugify(b['subject'])}_{b['page_nums'][0]}"
            b['page_number'] = b['page_nums'][0]
            b['context'] = b['subject']
            
        return bundles

    # [IN llm_service.py]

    def generate_batch(self, groups, prompt_template, progress_callback=None):
        raw_cards = []
        
        # [FIX 1] Retrieve Batch Size
        batch_size = int(self.settings.get("creator_batch_size", 1))
        if batch_size < 1: batch_size = 1
        
        # [FIX 2] Chunk the Groups
        chunks = [groups[i:i + batch_size] for i in range(0, len(groups), batch_size)]
        total_chunks = len(chunks)
        
        from prompts import MULTI_CARD_SCHEMA, SINGLE_CARD_SCHEMA
        
        for i, chunk in enumerate(chunks):
            if progress_callback: progress_callback(i + 1, total_chunks)
            
            # Prepare the text block for the prompt
            bundles_text = ""
            for grp in chunk:
                facts_str = "\n".join([f"  - {f['fact']}" for f in grp['facts']])
                b_id = grp.get('group_id', 'Unknown')
                subj = grp.get('subject', 'General')
                bundles_text += f"\n[ID: {b_id}]\nSUBJECT: {subj}\nFACTS:\n{facts_str}\n----------------\n"

            # Select Schema & Context based on Batch Size
            if batch_size > 1:
                current_schema = MULTI_CARD_SCHEMA
                prompt = prompt_template.format(bundles_block=bundles_text)
            else:
                current_schema = SINGLE_CARD_SCHEMA
                grp = chunk[0]
                facts_str = "\n".join([f"  - {f['fact']}" for f in grp['facts']])
                prompt = prompt_template.format(
                    facts_block=facts_str, 
                    text=facts_str,
                    topic=grp.get('subject', 'Medical Concept'),
                    bundle_id=str(grp.get('group_id', '1')),
                    subject=grp.get('subject', 'Medical Concept'),
                    context=grp.get('subject', 'Medical Concept')
                )

            # 3. Generation Loop (UPDATED RETRY LOGIC)
            # range(3) = Attempt 1, Retry 1, Retry 2
            success = False
            for attempt in range(3): 
                try:
                    response = self._call_llm(
                        [{"role": "user", "content": prompt}], 
                        temperature=0.2,
                        schema=current_schema
                    )
                    
                    if isinstance(response, dict):
                        data = response
                    else:
                        data = self._robust_json_parse(response)

                    if data:
                        new_cards_data = []
                        
                        # Case A: Multi-Card Response
                        if "cards" in data and isinstance(data["cards"], list):
                            new_cards_data = data["cards"]
                            
                        # Case B: Single-Card Response
                        elif "Front" in data and "Back" in data:
                            new_cards_data = [data]
                            
                        for card_item in new_cards_data:
                            if "Front" in card_item and "Back" in card_item:
                                # Recover Page Numbers
                                target_grp = next((g for g in chunk if str(g.get('group_id')) == str(card_item.get('Bundle_ID'))), None)
                                
                                if not target_grp and len(chunk) == len(new_cards_data):
                                     target_idx = new_cards_data.index(card_item)
                                     target_grp = chunk[target_idx]

                                if not target_grp: target_grp = chunk[0]
                                
                                p_nums = sorted(list(set(f['page_num'] for f in target_grp['facts'])))
                                
                                tool_call = {
                                    "name": "create_anki_card",
                                    "args": card_item
                                }
                                tool_call['args']['Page_numbers'] = p_nums
                                
                                raw_cards.append(tool_call)
                                success = True
                                
                        if success: break
                        
                except Exception as e:
                    self.log(f"   > [Creator] Error in batch {i+1} (Attempt {attempt+1}): {e}")
            
            if not success:
                self.log(f"❌ [Creator] Failed to generate batch {i+1} after 3 attempts.")
        
        return raw_cards

    def generate_cloze_batch(self, groups_list):
        self.log(f"   > [Cloze] Processing {len(groups_list)} cloze lists...")
        output_cards = []
        target_count = int(self.settings.get("cloze_keyword_count", 5))
        for group in groups_list:
            facts_str = "\n".join([f"  - {f['fact']}" for f in group['facts']])
            prompt = CREATOR_CLOZE_PARAGRAPH_PROMPT.format(
                subject=group.get('subject'),
                facts_block=facts_str,
                count=target_count,
            )
            result = self._call_llm(
                [{"role": "user", "content": prompt}], 
                temperature=0.1,
                schema=CLOZE_PARAGRAPH_SCHEMA
            )
            if result:
                output_cards.append({"name": "create_cloze_card", "args": result})
        return output_cards

    def generate_cloze_paragraph_batch(self, groups, progress_callback=None):
        self.log(f"   > [Creator] Generating {len(groups)} Cloze Paragraph cards...")
        
        target_count = int(self.settings.get("cloze_keyword_count", 5))
        generated_cards = []
        
        for i, grp in enumerate(groups):
            # [NEW] Update Progress
            if progress_callback:
                progress_callback(i + 1, len(groups))

            # ... (Rest of loop logic matches existing code) ...
            facts_text = "\n".join([f"- {f['fact']}" for f in grp['facts']])
            facts_block = "\n".join([f"  - {f['fact']}" for f in grp['facts']])
            p_nums = sorted(list(set(f['page_num'] for f in grp['facts'])))
            
            try:
                prompt_template = self.prompt_cloze_paragraph
                prompt_content = prompt_template.format(
                    facts=facts_text,
                    count=target_count,
                    bundle_id=grp.get('group_id', 'Unknown'),
                    subject=grp.get('subject', 'General'),
                    context=grp.get('context', 'General'),
                    facts_block=facts_block
                )
            except KeyError as e:
                self.log(f"   > [Creator Error] Prompt formatting failed: {e}")
                continue

            response = self._call_llm(
                messages=[{"role": "user", "content": prompt_content}], 
                temperature=0.7, 
                schema=CLOZE_PARAGRAPH_SCHEMA,
                json_mode=True
            )
            
            if response:
                topic = response.get("Topic", "General Concept")
                prose = response.get("Prose", "")
                keywords = response.get("Keywords", [])
                search_query = response.get("Search_Query", "")

                # --- INSERT THIS BLOCK HERE ---
                # [FIX] Fallback: If JSON Keywords are empty, extract from markdown bolding (**word**)
                if not keywords and "**" in prose:
                    # Extract text between double asterisks
                    extracted = re.findall(r'\*\*(.*?)\*\*', prose)
                    if extracted:
                        keywords = extracted
                        # Optional: Clean stars from prose so we don't get double bolding (**{{c1::...}}**)
                        prose = prose.replace("**", "")
                # ------------------------------

                generated_cards.append({
                    "name": "create_cloze_card",
                    "args": {
                        "Topic": topic,
                        "Prose": prose,
                        "Keywords": keywords,
                        "Search_Query": search_query,
                        "Page_numbers": p_nums
                    }
                })
            else:
                self.log(f"   > [Creator] Failed to generate card {i+1}/{len(groups)}")
                
        return generated_cards

    def run_critic(self, topic, old_question, new_back):
        """
        Rewrites a card's question to match a specific subset of the answer.
        """
        try:
            # [FIX] Provide ALL potential keys to prevent KeyError
            # The error 'new_back' implies the template uses {new_back}
            prompt = CRITIC_PROMPT_TEMPLATE.format(
                topic=topic,
                original_question=old_question, # Map for {original_question}
                old_question=old_question,      # Map for {old_question} (just in case)
                new_context=new_back,           # Map for {new_context}
                new_back=new_back               # Map for {new_back} (The fix)
            )
            
            # 1. Call LLM
            response = self._call_llm(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                schema=CRITIC_SCHEMA
            )
            
            # 2. Extract Result
            # Handle Dict (Schema Mode)
            if isinstance(response, dict):
                return response.get("refined_question", old_question)
            
            # Handle String (Fallback Mode)
            parsed = self._robust_json_parse(response)
            if parsed and isinstance(parsed, dict) and "refined_question" in parsed:
                return parsed["refined_question"]
                
            return old_question
            
        except Exception as e:
            self.log(f"   > [Critic] Error refining question: {e}")
            return old_question
        
    def _get_embeddings(self, texts):
        """
        Generates vector embeddings using all-mpnet-base-v2.
        """
        if not texts: return []
        
        # Lazy-load the model
        if not hasattr(self, '_embedder'):
            try:
                from sentence_transformers import SentenceTransformer
                # [FIX] Using the stronger model per your request
                self.log("   > [System] Loading embedding model (all-mpnet-base-v2)...")
                self._embedder = SentenceTransformer('all-mpnet-base-v2')
            except ImportError:
                self.log("❌ Error: 'sentence-transformers' not installed.")
                raise ImportError("Please run: pip install sentence-transformers")
            except Exception as e:
                self.log(f"❌ Error loading embedding model: {e}")
                raise e

        # Encode
        try:
            embeddings = self._embedder.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            self.log(f"❌ Error generating embeddings: {e}")
            return []

    def apply_safe_sieve(self, facts, objectives, threshold=0.25):
        """
        Filters facts against objectives using vector similarity.
        Returns ONLY the facts that match at least one objective.
        """
        if not facts or not objectives:
            return facts

        # 1. Embed Objectives & Facts
        # (Assuming self._get_embeddings is your internal helper)
        obj_embeddings = self._get_embeddings(objectives)
        fact_texts = [f['fact'] for f in facts]
        fact_embeddings = self._get_embeddings(fact_texts)
        
        kept_facts = []
        
        # 2. Compare
        import numpy as np
        # Convert to numpy for fast dot product
        v_facts = np.array(fact_embeddings)
        v_objs = np.array(obj_embeddings)
        
        # Normalize vectors to ensure dot product = cosine similarity
        norm_facts = np.linalg.norm(v_facts, axis=1, keepdims=True)
        norm_objs = np.linalg.norm(v_objs, axis=1, keepdims=True)
        v_facts = v_facts / (norm_facts + 1e-9)
        v_objs = v_objs / (norm_objs + 1e-9)
        
        # Similarity Matrix (Facts x Objectives)
        # Result is [num_facts, num_objectives]
        sim_matrix = np.dot(v_facts, v_objs.T)
        
        # 3. Filter
        for i, fact in enumerate(facts):
            # Get max similarity for this fact against ANY objective
            max_score = np.max(sim_matrix[i])
            
            if max_score >= threshold:
                kept_facts.append(fact)
            # else:
            #    print(f"Dropped Fact: {fact['fact'][:50]}... (Score: {max_score:.2f})")
                
        return kept_facts
        
    # [Add this new method to LLMService class]
    def run_goldenizer(self, context_block):
        """Refines raw Python-found objectives into a tight list of high-level goals."""
        
        # [FIX] New Prompt: Prioritizes SPECIFICITY over abstraction.
        prompt = f"""You are a medical data analyst.
                Below is a list of potential learning objectives extracted via Regex from a lecture.
                
                Your Task: Clean and deduplicate this list.
                
                CRITICAL RULES:
                1. **PRESERVE SPECIFICS:** Do NOT generalize. If the objective is "Explain the mechanism of Lisinopril", keep it. Do NOT rewrite it as "Explain drug mechanisms."
                2. **Deduplicate:** Only merge items that are identical in meaning.
                3. **Format:** Ensure every objective starts with a strong active verb (e.g., Analyze, Describe, Identify).
                4. **Scope:** Keep the list comprehensive. It is better to have 20 specific objectives than 5 vague ones.

                Input Data:
                {context_block}

                Output ONLY a JSON object with this schema:
                {{
                    "golden_objectives": [
                        "Specific Objective 1",
                        "Specific Objective 2"
                    ]
                }}
                """
        
        schema = {
            "type": "object",
            "properties": {
                "golden_objectives": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["golden_objectives"]
        }

        try:
            response = self._call_llm(
                [{"role": "user", "content": prompt}], 
                temperature=0.1, # [FIX] Lower temperature for stricter adherence
                schema=schema,
                json_mode=True
            )
            
            if response and "golden_objectives" in response:
                return [obj.strip().strip('.').strip() for obj in response["golden_objectives"]]
            
            # Fallback if JSON fails: just return what we put in (cleaned up)
            return [line.strip('- ') for line in context_block.split('\n') if line.strip().startswith('-')][:20]

        except Exception as e:
            self.log(f"   > [Goldenizer Error] Failed to refine objectives: {e}")
            # Fallback: return top 20 raw ones if AI fails
            raw_lines = [line.strip('- ') for line in context_block.split('\n') if line.strip().startswith('-')]
            return raw_lines[:20]

    def unload_model(self):
        """Force unloads the model from VRAM by setting keep_alive to 0"""
        
        if "Ollama" in self.provider:
            try:
                # Sending keep_alive: 0 tells Ollama to free VRAM immediately
                requests.post(
                    f"{self.api_base}/generate", 
                    json={"model": self.model_name, "keep_alive": 0},
                    timeout=2
                )
                self.log("   > [System] AI Model unloaded from VRAM.")
            except:
                pass