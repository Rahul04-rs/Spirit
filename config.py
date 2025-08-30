# backend/config.py

# Chunking settings
CHUNK_SIZE = 250       # words per chunk
CHUNK_OVERLAP = 50     # words overlap

# HuggingFace Granite Model
#GRANITE_MODEL = "ibm/granite-mistral-8x7b-instruct"
GRANITE_MODEL = "distilgpt2"
MAX_TOKENS = 300
TEMPERATURE = 0.7
