# src/config.py
import os

# --- PATHS ---
# Get the root directory of the project
# This assumes the script is run from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Use the Tomasulo path for now; we'll switch to gem5 later
CODEBASE_PATH = os.path.join(PROJECT_ROOT, "data/TomasuloAlgorithmImplementation/")
DB_PATH = os.path.join(PROJECT_ROOT, "db/")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env") # Path to the .env file

# --- MODEL CONFIGURATION ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'gemini-2.5-flash' # Using 1.5-flash for speed and cost

# --- DATABASE CONFIGURATION ---
COLLECTION_NAME = "tomasulo_code"

# --- CHUNKING CONFIGURATION ---
CHUNK_SIZE_LINES = 40
OVERLAP_LINES = 10