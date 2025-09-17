# src/config.py
import os

# --- PATHS ---
# Get the root directory of the project
# This assumes the script is run from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# gem5 code base root path
CODEBASE_PATH = os.path.join(PROJECT_ROOT, "data/gem5/")
DB_PATH = os.path.join(PROJECT_ROOT, "db/")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env") # Path to the .env file

# --- MODEL CONFIGURATION ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'gemini-2.5-flash' # Using 1.5-flash for speed and cost

# --- DATABASE CONFIGURATION ---
COLLECTION_NAME = "gem5_code_ast"