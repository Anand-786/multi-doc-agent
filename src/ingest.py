import os
import json
import clang.cindex
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import multiprocessing

# Import configuration
from config import CODEBASE_PATH, DB_PATH, EMBEDDING_MODEL_NAME, COLLECTION_NAME

# Node kinds to extract from the AST
NODE_KINDS_TO_EXTRACT = [
    clang.cindex.CursorKind.FUNCTION_DECL,
    clang.cindex.CursorKind.CXX_METHOD,
    clang.cindex.CursorKind.CLASS_DECL,
    clang.cindex.CursorKind.STRUCT_DECL,
]

def load_compilation_database(codebase_path: str) -> dict:
    """
    Loads the compile_commands.json file and creates a lookup dictionary.

    Returns:
        A dictionary mapping file paths to their compilation arguments.
    """
    compdb_path = os.path.join(codebase_path, 'compile_commands.json')
    if not os.path.exists(compdb_path):
        print(f"❌ ERROR: compile_commands.json not found at '{compdb_path}'")
        return None
        
    with open(compdb_path, 'r') as f:
        compilation_data = json.load(f)
    
    # Create a dictionary for fast lookups
    db = {entry['file']: entry['arguments'] for entry in compilation_data}
    print(f"✅ Loaded compilation database with {len(db)} entries.")
    return db

def extract_chunks_from_ast(cursor: clang.cindex.Cursor, file_content: str, target_filename: str) -> list[dict]:
    """
    Recursively traverses the AST and extracts semantically complete chunks,
    staying within the target file.
    """
    chunks = []
    try:
        if cursor.location.file and os.path.samefile(cursor.location.file.name, target_filename):
            if cursor.kind in NODE_KINDS_TO_EXTRACT:
                start_offset = cursor.extent.start.offset
                end_offset = cursor.extent.end.offset
                code_text = file_content[start_offset:end_offset]

                chunks.append({
                    "text": code_text,
                    "metadata": {
                        "file_path": cursor.location.file.name,
                        "type": cursor.kind.name,
                        "name": cursor.spelling,
                        "start_line": cursor.extent.start.line,
                        "end_line": cursor.extent.end.line
                    }
                })
            for child in cursor.get_children():
                chunks.extend(extract_chunks_from_ast(child, file_content, target_filename))
    except FileNotFoundError:
        pass
    return chunks

def find_cpp_files(directory: str) -> list[str]:
    """Finds all .cc and .hh files in a directory."""
    cpp_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".cc", ".hh")):
                cpp_files.append(os.path.join(root, file))
    return cpp_files

def parse_file(file_path: str, comp_db: dict) -> list[dict]:
    """Parses a single C++ file using its specific compilation commands."""
    try:
        # We only care about files present in the compilation database
        if file_path not in comp_db:
            return []

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()

        # Get the specific arguments for this file from the database
        # We slice [1:] to remove the compiler executable itself (e.g., 'g++')
        args = comp_db[file_path][1:]

        index = clang.cindex.Index.create()
        translation_unit = index.parse(file_path, args=args)
        
        # Check for fatal errors that would prevent a valid AST
        fatal_errors = [d for d in translation_unit.diagnostics if d.severity >= d.Error]
        if fatal_errors:
            # We can log these, but for now we'll just skip the file on fatal error
            return []
            
        return extract_chunks_from_ast(translation_unit.cursor, file_content, file_path)
    except Exception:
        # A catch-all for any other unexpected parsing issues
        return []

def run_ingestion():
    """Main function to run the entire high-fidelity ingestion pipeline."""
    
    print("--- Step 1/5: Loading Compilation Database ---")
    comp_db = load_compilation_database(CODEBASE_PATH)
    if not comp_db:
        return

    print("\n--- Step 2/5: Discovering C++ Files ---")
    all_files = find_cpp_files(CODEBASE_PATH)
    print(f"✅ Found {len(all_files)} C++ source and header files to process.")

    print("\n--- Step 3/5: Parsing All Files with High Fidelity ---")
    all_chunks = []
    # We will only parse files that are actually in our compilation database
    files_to_parse = [f for f in all_files if f in comp_db]
    with tqdm(total=len(files_to_parse), desc="Parsing files") as pbar:
        for file_path in files_to_parse:
            chunks = parse_file(file_path, comp_db)
            all_chunks.extend(chunks)
            pbar.update(1)
    
    if not all_chunks:
        print("❌ ERROR: No chunks were extracted. Check the compilation database and parsing logic.")
        return
    print(f"✅ Extracted a total of {len(all_chunks)} semantic chunks.")

    print("\n--- Step 4/5: Initializing Vector Database & Embedding Model ---")
    client = chromadb.PersistentClient(path=DB_PATH)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Deleting existing collection '{COLLECTION_NAME}'.")
        client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"✅ ChromaDB and embedding model are ready.")

    print("\n--- Step 5/5: Generating Embeddings and Storing in DB ---")
    batch_size = 256
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches", total=total_batches):
        batch = all_chunks[i:i+batch_size]
        documents = [item['text'] for item in batch]
        metadatas = [item['metadata'] for item in batch]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        embeddings = embedding_model.encode(documents).tolist()
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"\n🎉 Ingestion complete! Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    run_ingestion()

