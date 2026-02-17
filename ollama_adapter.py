import requests
import json
import time
import re
import os
from itertools import groupby
import threading
import numpy as np

# --- API CONFIGURATION ---
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"

# --- MODEL CONFIGURATION ---
MODEL_EXTRACTION = "ministral-3:8b-instruct-2512-q4_K_M" 
MODEL_GENERATION = "ministral-3:8b-instruct-2512-q4_K_M" 

from prompts import (
    HARVESTER_SYSTEM_PROMPT, 
    LIBRARIAN_SYSTEM_PROMPT, 
    CREATOR_SINGLE_VIGNETTE_PROMPT,
    CLOZE_SINGLE_PROMPT,
    CRITIC_PROMPT_TEMPLATE,
    OBJECTIVE_FINDER_PROMPT
)

# --- STRICT SCHEMAS ---

SINGLE_CARD_SCHEMA = {
  "type": "object",
  "properties": {
    "Topic": {"type": "string"},
    "Front": {"type": "string"},
    "Back": {"type": "string"},
    "Search_Query": {"type": "string"}
  },
  "required": ["Front", "Back", "Search_Query"]
}

SINGLE_CLOZE_SCHEMA = {
  "type": "object",
  "properties": {
    "Header": {"type": "string"},
    "Text": {"type": "string"},
    "Search_Query": {"type": "string"}
  },
  "required": ["Header", "Text", "Search_Query"] # FIX: Made Search_Query Required
}

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "refined_question": {"type": "string"}
    },
    "required": ["refined_question"]
}

LIBRARIAN_STRICT_SCHEMA = {
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "tag": {
            "type": "string",
            "enum": ["KEEP", "NOISE"]
          }
        },
        "required": ["tag"]
      }
    }
  },
  "required": ["data"]
}

SCOUT_STRICT_SCHEMA = {
  "type": "object",
  "properties": {
    "topics": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["topics"]
}

OBJECTIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "objectives": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["objectives"]
}

OLLAMA_API_LOCK = threading.Lock()

class OllamaHandler:
    def __init__(self, logger_func):
        self.log = logger_func
        try:
            cpu_count = os.cpu_count() or 4
            self.threads = max(1, cpu_count - 2)
        except:
            self.threads = 4

    def unload_model(self, model_name):
        try:
            payload = {"model": model_name, "keep_alive": 0}
            requests.post(OLLAMA_GEN_URL, json=payload, timeout=10)
            self.log(f"   > [System] Unloaded {model_name} from memory.")
        except Exception as e:
            self.log(f"   > [System] Warning: Failed to unload model: {e}")

    def force_clear_memory(self):
        self.unload_model(MODEL_GENERATION)

    def _call_ollama_chat(self, messages, model, retries=0, json_mode=True, context_size=8192, schema=None, temperature=0.1):
        payload = {
            "model": model,
            "messages": messages, 
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": context_size,
                "num_thread": self.threads,
                "num_predict": 4096
            }
        }
        if schema:
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"

        for attempt in range(retries + 1):
            try:
                with OLLAMA_API_LOCK:
                    response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("message", {}).get("content", "")
                    if schema:
                        try: return json.loads(content)
                        except: return self._robust_json_parse(content)
                    if json_mode:
                        return self._robust_json_parse(content)
                    return content
                else:
                    self.log(f"   > [Ollama] Error {response.status_code}: {response.text}")
            except Exception as e:
                self.log(f"   > [Ollama] Connection Error: {e}")
            if attempt < retries:
                time.sleep(2)
        return None

    def extract_objectives(self, text_chunk):
        prompt = OBJECTIVE_FINDER_PROMPT.replace("{full_text}", text_chunk[:6000])
        messages = [{"role": "user", "content": prompt}]
        response = self._call_ollama_chat(messages, MODEL_EXTRACTION, temperature=0.1, schema=OBJECTIVE_SCHEMA)
        if response and "objectives" in response:
            return response["objectives"]
        return []

    def run_scout_summary(self, text_chunk):
        prompt = (
            "Analyze the following document excerpt and identify the Top 10 Core Topics.\n"
            "Return ONLY a JSON object with a 'topics' key containing the list of strings.\n\n"
            f"--- TEXT ---\n{text_chunk[:10000]}" 
        )
        messages = [{"role": "user", "content": prompt}]
        response = self._call_ollama_chat(messages, MODEL_GENERATION, temperature=0.3, schema=SCOUT_STRICT_SCHEMA)
        if response and "topics" in response:
            return response["topics"]
        return ["General Medical Concepts"]

    def run_harvester(self, text_chunk, page_num):
        pass

    def run_stateful_harvester(self, pages_data_list):
        if pages_data_list and isinstance(pages_data_list[0], dict):
             pages_data_list = [(p['page'], p['text']) for p in pages_data_list]
        elif pages_data_list and isinstance(pages_data_list[0], str):
             pages_data_list = [(1, pages_data_list[0])]

        all_facts = []
        previous_subject = "None"
        BATCH_SIZE = 3
        total_batches = (len(pages_data_list) + BATCH_SIZE - 1) // BATCH_SIZE
        
        self.log(f"   > [Harvester] Starting extraction on {len(pages_data_list)} pages (Batches: {total_batches})")

        for i in range(0, len(pages_data_list), BATCH_SIZE):
            chunk_tuples = pages_data_list[i : i + BATCH_SIZE]
            start_pg = chunk_tuples[0][0]
            end_pg = chunk_tuples[-1][0]
            chunk_text = ""
            for p_num, p_text in chunk_tuples:
                chunk_text += f"--- Page {p_num} ---\n{p_text}\n\n"
            
            prompt = HARVESTER_SYSTEM_PROMPT.format(
                previous_subject=previous_subject,
                text_chunk=chunk_text
            )
            SCHEMA = {
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
            response = self._call_ollama_chat(
                [{"role": "user", "content": prompt}], 
                MODEL_EXTRACTION,
                schema=SCHEMA
            )
            batch_count = 0
            if response and "data" in response:
                page_facts = response['data']
                batch_count = len(page_facts)
                primary_page = chunk_tuples[0][0] 
                for f in page_facts:
                    if 'page_num' not in f: f['page_num'] = primary_page
                all_facts.extend(page_facts)
                valid = [x['subject'] for x in page_facts if x['subject'] not in ["None", "Unknown", "General"]]
                if valid:
                    previous_subject = valid[-1] if len(valid[-1]) > 3 else "None"
            self.log(f"   > [Harvester] Batch {i//BATCH_SIZE + 1}/{total_batches} (Pages {start_pg}-{end_pg}): Found {batch_count} facts. (Context: {previous_subject})")
        return all_facts

    def run_librarian(self, facts_list):
        if not facts_list: return []
        self.log(f"   > [Librarian] Reviewing {len(facts_list)} facts...")
        facts_block = "\n".join([f"{i+1}. {f['fact']}" for i, f in enumerate(facts_list)])
        prompt = LIBRARIAN_SYSTEM_PROMPT.replace("{facts_block}", facts_block)
        response = self._call_ollama_chat(
            [{"role": "user", "content": prompt}], 
            MODEL_EXTRACTION, 
            temperature=0.2,
            schema=LIBRARIAN_STRICT_SCHEMA
        )
        cleaned_facts = []
        kept_count = 0
        removed_count = 0
        if response and "data" in response:
            tags = response["data"]
            for i, fact in enumerate(facts_list):
                tag_entry = tags[i] if i < len(tags) else {"tag": "KEEP"}
                tag = tag_entry.get("tag", "KEEP").upper()
                if tag != "NOISE":
                    fact['metadata'] = {"tag": "KEEP"}
                    cleaned_facts.append(fact)
                    kept_count += 1
                else:
                    removed_count += 1
        else:
            self.log("   > [Librarian] Failed to parse response. Keeping all facts.")
            return facts_list
        self.log(f"   > [Librarian] Done. Kept: {kept_count} | Removed: {removed_count} (Noise)")
        return cleaned_facts

    def group_facts_by_topic(self, tagged_facts):
        def slugify(text):
            clean = re.sub(r'[^\w\s-]', '', text).strip().lower()
            return re.sub(r'[-\s]+', '_', clean)
        self.log(f"   > [Assembly] Starting Assembly with {len(tagged_facts)} facts.")
        indexed_facts = []
        for i, f in enumerate(tagged_facts):
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
        for (pg, subj), facts in micro_bundles.items():
            chunks = [facts[i:i+4] for i in range(0, len(facts), 4)]
            for chunk in chunks:
                bundles.append({
                    "subject": subj, "Subject": subj,
                    "page_nums": [pg], "Page_Nums": [pg],
                    "facts": chunk, "Facts": chunk,
                    "centroid": None
                })
        # Vertical
        bundles.sort(key=lambda x: (x['subject'], x['page_nums'][0]))
        merged_v = []
        if bundles:
            curr = bundles[0]
            for next_b in bundles[1:]:
                is_adj = abs(curr['page_nums'][-1] - next_b['page_nums'][0]) <= 1
                if (curr['subject'] == next_b['subject'] and is_adj and len(curr['facts']) + len(next_b['facts']) <= 5):
                    curr['facts'] += next_b['facts']
                    curr['page_nums'] = sorted(list(set(curr['page_nums'] + next_b['page_nums'])))
                    continue
                merged_v.append(curr)
                curr = next_b
            merged_v.append(curr)
        bundles = merged_v
        # Horizontal
        bundles.sort(key=lambda x: x['page_nums'][0])
        merged_h = []
        if bundles:
            curr = bundles[0]
            for next_b in bundles[1:]:
                is_same_page = curr['page_nums'][0] == next_b['page_nums'][0]
                if (is_same_page and len(curr['facts']) < 3 and len(curr['facts']) + len(next_b['facts']) <= 5):
                    curr['facts'] += next_b['facts']
                    if curr['subject'] != next_b['subject']:
                         curr['subject'] = f"{curr['subject']} & {next_b['subject']}"
                    continue
                merged_h.append(curr)
                curr = next_b
            merged_h.append(curr)
        bundles = merged_h
        # Vector
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
                if len(curr['facts']) >= 3:
                    final_pass.append(curr)
                    continue
                best_idx, best_score = -1, -1
                for i, candidate in enumerate(bundles):
                    if len(curr['facts']) + len(candidate['facts']) > 5: continue
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
        except: pass
        for b in bundles:
            b['group_id'] = f"{slugify(b['subject'])}_{b['page_nums'][0]}"
            b['page_number'] = b['page_nums'][0]
            b['context'] = b['subject']
        return bundles

    def generate_batch(self, groups_list, full_prompt_template):
        self.log(f"   > [Creator] Processing {len(groups_list)} bundles sequentially (1-by-1)...")
        output_cards = []
        for i, group in enumerate(groups_list):
            facts_str = "\n".join([f"  - {f['fact']}" for f in group['facts']])
            prompt = CREATOR_SINGLE_VIGNETTE_PROMPT.format(
                bundle_id=group.get('group_id'),
                subject=group.get('subject'),
                context=group.get('context'),
                facts_block=facts_str
            )
            result = self._call_ollama_chat(
                [{"role": "user", "content": prompt}], 
                MODEL_GENERATION, 
                temperature=0.1,
                json_mode=True,
                schema=SINGLE_CARD_SCHEMA
            )
            if result:
                result['Page_numbers'] = group.get('page_nums', [1])
                result['Sub_Topics'] = []
                result['Exhaustive_Question'] = result.get('Front')
                output_cards.append({"name": "create_anki_card", "args": result})
        self.log(f"   > [Creator] Successfully created {len(output_cards)} cards.")
        return output_cards

    def generate_cloze_batch(self, groups_list):
        self.log(f"   > [Cloze] Processing {len(groups_list)} cloze lists sequentially...")
        output_cards = []
        for i, group in enumerate(groups_list):
            facts_str = "\n".join([f"  - {f['fact']}" for f in group['facts']])
            prompt = CLOZE_SINGLE_PROMPT.format(
                subject=group.get('subject'),
                facts_block=facts_str
            )
            result = self._call_ollama_chat(
                [{"role": "user", "content": prompt}], 
                MODEL_GENERATION, 
                temperature=0.1,
                json_mode=True,
                schema=SINGLE_CLOZE_SCHEMA
            )
            if result:
                output_cards.append({"name": "create_cloze_card", "args": result})
        self.log(f"   > [Cloze] Created {len(output_cards)} cloze cards.")
        return output_cards

    def run_critic(self, topic, old_question, new_back):
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            topic=topic, 
            original_question=old_question, 
            new_back=new_back
        )
        response = self._call_ollama_chat(
            [{"role": "user", "content": prompt}], 
            MODEL_GENERATION, 
            temperature=0.2, 
            schema=CRITIC_SCHEMA
        )
        if response and "refined_question" in response:
            return response["refined_question"]
        return f"{topic}: {old_question}"

    def _robust_json_parse(self, text, schema=None):
        if not text: return None
        text = text.strip()
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1).strip())
            except: pass
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except: pass
        return None

    def apply_safe_sieve(self, facts, golden_objectives):
        if not golden_objectives: return facts
        self.log(f"   > [Sieve] Filtering {len(facts)} facts against {len(golden_objectives)} Golden Objectives...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            obj_vecs = model.encode(golden_objectives)
            shielded = []
            for f in facts:
                f_vec = model.encode([f['fact']])[0]
                scores = np.dot(obj_vecs, f_vec.T)
                if np.max(scores) > 0.25:
                    shielded.append(f)
            self.log(f"   > [Sieve] Retained {len(shielded)} facts (Dropped {len(facts) - len(shielded)})")
            return shielded
        except Exception as e:
            self.log(f"   > [Sieve] Error: {e}")
            return facts