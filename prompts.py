# --- PROMPTS ---

HARVESTER_SYSTEM_PROMPT = """
Task: Extract all facts from the provided text.

**INSTRUCTIONS:**
1. **SELF-CONTAINED FACTS:**
   - Autonomy: Each fact must be a complete, self-contained sentence. Facts should make sense on their own without requiring additional context.
   - No Pronouns: Avoid using pronouns like "it", "they", "this", etc. Each fact should clearly state the subject.
   - Lists: For simple lists, create a single fact that includes the entire list. For complex lists with explanations, break them into multiple facts.
   - Text Adherence - IMPERATIVE: Do NOT add any external knowledge or information. ONLY use the information explicitly provided in the text.
   - Flatten Tables: If the text contains tables, convert each row or column into individual facts.
   - Comprehensiveness: Extract ALL unique facts. Do NOT omit any information. Do NOT describe what the text "talks about" or "includes", INSTEAD extract the information into discrete facts.
2. **PAGE TRACKING:**
   - The text contains markers like `--- Page 5 ---`.
   - Assign the correct `page_num` to every fact.

**OUTPUT SCHEMA (JSON):**
{{
  "data": [
    {{
      "subject": "String (Specific Noun)",
      "fact": "String (Full Sentence)",
      "page_num": "Integer"
    }}
  ]
}}

**TEXT TO PROCESS:**
{text_chunk}
"""
# [NEW] - Objective Verifier
CREATOR_OBJECTIVE_PROMPT = """
**INPUT DATA:**
- Topic: {topic}
**FACTS:**
{facts_block}

**INSTRUCTIONS:**
1. **Goal:** The card must test whether the student understands the "facts". 
- ONLY test the facts.
- Do NOT add external knowledge. 
2. **Front (Question):**
   - Write a specific question asking for the core concept of the facts.
   - The question should be as short, simple and concise as possible.
3. **Back (Answer):**
   - Synthesize the FACTS into a concise answer.
   - Use bullet points and sub bullet points for clarity.
   - Use an outline format.
   - The answer should be as short, simple and concise as possible.
4. **Search Query:** A 2-3 word visual search term.

**REQUIRED JSON OUTPUT:**
{{
    "Topic": "String (The Objective Name)",
    "Front": "String (The Question)",
    "Back": "String (The Answer)",
    "Search_Query": "String"
}}
"""

# [NEW] - The Judge
MAPPER_JUDGE_PROMPT = """
You are a classifier. Match the FACT to the best OBJECTIVE option.

FACT: "{fact}"

OPTIONS:
{options_list}

INSTRUCTIONS:
- Return the ID number (0, 1, or 2) of the best match.
- If it matches NONE, return -1.

JSON OUTPUT:
{{
  "best_match_id": Integer
}}
"""

LIBRARIAN_SYSTEM_PROMPT = """
Task: Classify medical statements ONLY as NOISE or KEEP.

**TAG DEFINITIONS:**
1. **NOISE**: Useless text, headers, footers, page numbers, navigational text, "email me at...", reference sections, repeated titles, or administrative fluff.
2. **KEEP**: Any statement that contains substantive factual information relevant to the subject.

**INSTRUCTIONS:**
- You must output either "NOISE" or "KEEP".
- Be conservative. If a fact has any yield, keep it. Only tag meta-text or junk as NOISE.

**INPUT (JSON):**
{facts_json}
"""

GOLDEN_OBJECTIVE_PROMPT = """
You are a medical curriculum designer. I will provide a list of Raw Learning Objectives and a list of specific Document Topics.
Rewrite the objectives to be specific to the topics. The goal is to break the objectives into their constituent components in relation to each of the topics.
***MANDATE*** Do NOT reduce the SCOPE of the objectives.
***IDEAL*** You want to directly specify each scope component of each and every objective, inserting topics where appropriate.
1. Map every raw objective to a Topic. Do not skip any objectives. You are encouraged to split objectives into multiple parts if necessary. Especially split listed objectives into their parts.
2. Be specific. When a series of topics is applicable to an objective, provide both the original objective text and versions of the objective with the topics incorporated.
3. Be comprehensive. Ensure your refined objective list comprehensively covers the intended scope of the original topics and objectives. Attempt to incorporate EVERY topic.
4. Output a simple JSON list of strings.

---START OF EXAMPLE INPUT AND OUTPUT---
Topic - "Alcohol intake. Alcoholic Pancreatitis. Alcohol abuse."
Objective - "Describe the physical and metabolic consequences of excessive alcohol intake and identify the signs and effects of chronic alcohol use"
Refined Objectives - "Describe the physical consequences of excessive alcohol intake.
Describe the metabolic consequences of excessive alcohol intake.
Describe the metabolic consequences of excessive alcohol intake and alcoholic pancreatitis.
Identify the signs of chronic alcohol use.
Identfy the effects of chronic alcohol use."
---END EXAMPLE---

--START OF ACTUAL DATA---
TOPICS: {topics}
RAW OBJECTIVES:
{raw_objectives}
"""

OBJECTIVE_FINDER_PROMPT = """
You are an expert document analyst. Your sole task is to find the "Learning Objectives" section in the text.

RULES:
1.  Scan the text for a section explicitly titled "Learning Objectives", "Objectives", "Session Goals", or similar.
2.  If found, return the list of objectives as a JSON array of strings.
3.  If NOT found, return an empty list.

--- DOCUMENT TEXT ---
{full_text}
"""

CREATOR_SINGLE_VIGNETTE_PROMPT = """
You are a medical school professor. Create EXACTLY ONE High-Yield Anki Card from the provided Concept Bundle.

INPUT BUNDLE:
[ID: {bundle_id}]
SUBJECT: {subject}
CONTEXT: {context}
FACTS:
{facts_block}

INSTRUCTIONS:
1. **Analyze the Facts:** - Review all facts carefully.
    - Identify the core concept that ties them together.

2. **FRONT (Question):**
   - Write a clear, specific question that targets the core concept.
   - Ensure the question is EXHAUSTIVE, covering ALL aspects of the answer.
   - Do NOT reveal the answer in the question.
   - Keep it concise.

3. **BACK (Answer):**
   - Provide a concise, outline style response containing ALL facts given.
   - The format MUST be a bullet point list.
   - Be precise and avoid fluff.
   - Be as simple and concise as possible without losing ANY information from the facts.

4. **SEARCH QUERY:**
   - Provide the main subject as text for a visual search.

**REQUIRED JSON OUTPUT:**
{{
    "Topic": "String (The core subject)",
    "Front": "String (The question)",
    "Back": "String (The concise answer)",
    "Search_Query": "String"
}}
"""

# --- NEW: BATCH CREATOR PROMPT ---
CREATOR_BATCH_PROMPT = """
You are a professor.
For EACH bundle, create ONE High-Yield Anki Card.

**INPUT BUNDLES:**
{bundles_block}

**INSTRUCTIONS FOR EACH CARD:**
1. **Analyze the Facts:**
    - Identify the core concept that ties them together.

2. **FRONT (Question):**
   - Write a clear, specific question that targets the core concept.
   - Ensure the question is EXHAUSTIVE, covering ALL aspects of the answer.
   - Do NOT reveal the answer in the question.
   - Keep it concise.

3. **BACK (Answer):**
   - Summarize all facts in a concise, outline style response using bullet points.
   - You MUST use bullet points or sub bullet points on every new line.
   - Be comprehensive, but be as concise and short as possible without losing ANY information from the facts.
   
4. **Search Query:** A visual search term.
   - Provide the main subject as text for a visual search.

**REQUIRED JSON OUTPUT:**
{{
  "cards": [
    {{
      "Bundle_ID": "String (Match the ID from input)",
      "Topic": "String",
      "Front": "String",
      "Back": "String",
      "Search_Query": "String"
    }}
  ]
}}
"""

CLOZE_SINGLE_PROMPT = """
You are an expert medical educator. Convert the following list of facts into a **Single Cloze Deletion Card**.

--- CONTEXT ---
Header: {subject}
Facts:
{facts_block}

--- INSTRUCTIONS ---
1. **Header:** Create a clear title for the list.
2. **Text:** Present the facts as a bulleted list.
3. **Cloze:** Apply Anki cloze syntax {{c1::answer}} to the most critical keyword in each bullet.
4. **Search Query:** A visual search term for this list.

**REQUIRED JSON OUTPUT:**
{{
  "Header": "String",
  "Text": "String (Bulleted list with clozes)",
  "Search_Query": "String"
}}
"""

CRITIC_PROMPT_TEMPLATE = """
You are an Anki Card Specialist. I have split a large flashcard into a smaller, more specific one.

--- INPUT DATA ---
1. **Original Topic:** {topic}
2. **Original Question:** {original_question}
3. **New Answer Subset:** {new_back}

--- TASK ---
Rewrite the "Original Question" so it specifically asks for the information in the "New Answer Subset".
   - Do NOT simply ask "What is ...?". Be specific based on the facts provided.
   - The question should be as short, simple and concise as possible.

**REQUIRED JSON OUTPUT:**
{{
    "refined_question": "String"
}}
"""

CREATOR_CLOZE_PARAGRAPH_PROMPT = """
You are a fact summarizer. The goal is to create sentences that could be used for "fill in the blank" style flashcards.

INSTRUCTIONS:
1. **Write Prose:** Synthesize the facts into sentence(s). Do NOT use bullet points.
2. **Identify Possible Blanks as "**bolded**"**
   - Identify {count} possible portions of the text that could be converted into blanks. Label these portions as **bolded** (With double **).
   - Bolded text should be: lists of symptoms, definitions, indications for, results of, characteristics, mechanisms, processes, sequences, or other high-yield information that describes the subject.
   - Bolded text should NOT be: common phrases, stop words, trivial information, or low-yield data.
   - Bolded text should ABSOLUTELY NOT be: the subject or main topic of the sentence.
   - AGAIN IMPERATIVE: THE SUBJECT OR MAIN TOPIC OF THE SENTENCE MUST NEVER BE BOLDED. ONLY BOLD INFORMATION EXPLAINING OR DESCRIBING THE SUBJECT.
   - Bolded text should be inferable from the context of the sentence after being "blanked". Do NOT blank/bold necessary context. 
- When choosing bolded text, ask yourself: "If this bolded portion were blanked out, could a student still understand the context of the sentence and infer the answer?"
3. **Fact adherence:** Only include information directly from the input data. Do NOT add external knowledge.
4. **Conciseness:** Include all facts, but be as short, simple, and concise as possible.

Example:
Cholelithiasis is characterized by the formation of **gallstones** in the gallbladder, which can lead to **biliary colic, cholecystitis, and jaundice**.

**INPUT DATA:**
{facts}

**REQUIRED JSON OUTPUT:**
{{
  "Topic": "A concise subject header",
  "Prose": "The sentence(s) with **bolded** keywords...",
  "Keywords": "List of bolded blanks/keywords",
  "Search_Query": "Visual search term"
}}
"""

# --- NEW: SCHEMA ---
CLOZE_PARAGRAPH_SCHEMA = {
  "type": "object",
  "properties": {
    "Topic": {"type": "string"},
    "Prose": {"type": "string"},
    "Keywords": {"type": "array", "items": {"type": "string"}},
    "Search_Query": {"type": "string"}
  },
  "required": ["Topic", "Prose", "Keywords", "Search_Query"]
}

# --- SCHEMAS ---

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

# --- NEW: MULTI CARD SCHEMA ---
MULTI_CARD_SCHEMA = {
  "type": "object",
  "properties": {
    "cards": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "Bundle_ID": {"type": "string"},
          "Topic": {"type": "string"},
          "Front": {"type": "string"},
          "Back": {"type": "string"},
          "Search_Query": {"type": "string"}
        },
        "required": ["Bundle_ID", "Front", "Back", "Search_Query"]
      }
    }
  },
  "required": ["cards"]
}

SINGLE_CLOZE_SCHEMA = {
  "type": "object",
  "properties": {
    "Header": {"type": "string"},
    "Text": {"type": "string"},
    "Search_Query": {"type": "string"}
  },
  "required": ["Header", "Text", "Search_Query"]
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

# Compatibility placeholders
BUILDER_PROMPT = "Placeholder" 
CLOZE_BUILDER_PROMPT = "Placeholder" 
CONTEXTUAL_CLOZE_BUILDER_PROMPT = "Placeholder"
MERMAID_BUILDER_PROMPT = "Placeholder"
VIGNETTE_INSTRUCTION = "Placeholder"
HARVESTER_SCHEMA = """{"data": [{"subject": "String", "fact": "String", "page_num": "Integer"}]}"""
LIBRARIAN_SCHEMA = """{"data": [{"tag": "String (NOISE or KEEP)"}]}"""
CREATOR_SCHEMA = """{"cards": [{"Front": "String", "Back": "String"}]}"""