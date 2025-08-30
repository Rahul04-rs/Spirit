# backend/rag.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from backend.config import GRANITE_MODEL, MAX_TOKENS, TEMPERATURE

# Load IBM Granite model via HuggingFace
#tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL)
#model = AutoModelForCausalLM.from_pretrained(GRANITE_MODEL, torch_dtype=torch.float16, device_map="auto")
#tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL)
#model = AutoModelForCausalLM.from_pretrained(GRANITE_MODEL, torch_dtype=torch.float16, device_map="auto")
#GRANITE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",

#model = AutoModelForCausalLM.from_pretrained(
 #   GRANITE_MODEL,
  #  torch_dtype=torch.float32,
   # device_map=None
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Step 1: Set the model name (do NOT use it as a keyword argument)
GRANITE_MODEL = "distilgpt2"

# ✅ Step 2: Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    GRANITE_MODEL,
    torch_dtype=torch.float32,  # Safe for CPU
    device_map=None             # Don't try to use GPU
)


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_TOKENS,
    temperature=TEMPERATURE
)

def compose_prompt(query: str, retrieved_chunks: list) -> str:
    context = "\n\n".join([f"[Source {i+1}] {c['text']}" for i, c in enumerate(retrieved_chunks)])
    prompt = f"""
You are StudyMate, an AI academic tutor. Answer the question using ONLY the provided sources.

Sources:
{context}

Question:
{query}

Answer concisely, and mention the sources used.
"""
    return prompt

def generate_answer(prompt: str) -> str:
    """Generate answer using IBM Granite via HuggingFace pipeline."""
    result = generator(prompt)
    return result[0]["generated_text"]
