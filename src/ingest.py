import os
import clang.cindex
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import multiprocessing

# Import configuration from our config file
# IMPORTANT: Update your config.py to point CODEBASE_PATH to your gem5 directory
# e.g., CODEBASE_PATH = os.path.join(PROJECT_ROOT, "data/gem5/")
from config import CODEBASE_PATH, DB_PATH, EMBEDDING_MODEL_NAME, COLLECTION_NAME

# We are interested in these kinds of AST nodes.
NODE_KINDS_TO_EXTRACT = [
    clang.cindex.CursorKind.FUNCTION_DECL,
    clang.cindex.CursorKind.CXX_METHOD,
    clang.cindex.CursorKind.CLASS_DECL,
    clang.cindex.CursorKind.STRUCT_DECL,
]

def extract_chunks_from_ast(cursor: clang.cindex.Cursor, file_content: str, target_filename: str) -> list[dict]:
    """
    Recursively traverses the AST and extracts semantically complete chunks,
    using a robust file path comparison to stay within the target file.
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
        # This can happen with virtual files or other edge cases.
        pass
    return chunks

def find_cpp_files(directory: str) -> list[str]:
    """Walks a directory and finds all .cc and .hh files."""
    cpp_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".cc", ".hh")):
                cpp_files.append(os.path.join(root, file))
    return cpp_files

def parse_file(file_path: str, include_paths: list[str]) -> list[dict]:
    """Parses a single C++ file and returns a list of chunks."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()

        # Construct compiler arguments, including the C++ standard and include paths
        args = ['-std=c++11'] + [f'-I{path}' for path in include_paths]

        index = clang.cindex.Index.create()
        translation_unit = index.parse(file_path, args=args)

        # Check for parsing errors
        parsing_errors = [d for d in translation_unit.diagnostics if d.severity >= d.Error]
        if parsing_errors:
            # Log a warning but still attempt to extract chunks, as some might be valid
            print(f"  - WARNING: Found {len(parsing_errors)} parsing errors in {os.path.basename(file_path)}. First error: {parsing_errors[0]}")

        return extract_chunks_from_ast(translation_unit.cursor, file_content, file_path)
    except Exception as e:
        print(f"  - WARNING: Failed to parse {os.path.basename(file_path)}. Error: {e}")
        return []

def run_ingestion():
    """Main function to run the entire ingestion pipeline for the gem5 codebase."""
    
    print("--- Step 1/5: Discovering C++ Files ---")
    if not os.path.exists(CODEBASE_PATH):
        print(f"❌ ERROR: Codebase directory not found at '{CODEBASE_PATH}'.")
        print("   Please update the CODEBASE_PATH in 'src/config.py'.")
        return
        
    all_files = find_cpp_files(CODEBASE_PATH)
    if not all_files:
        print("❌ ERROR: No .cc or .hh files found in the codebase path.")
        return
    print(f"✅ Found {len(all_files)} C++ source and header files.")

    print("\n--- Step 2/5: Parsing All Files with AST ---")

    # --- CRITICAL FIX ---
    # Define the primary include path for gem5, which is the 'src' directory.
    # We also include the main codebase path itself.
    gem5_src_path = os.path.join(CODEBASE_PATH, 'src')
    include_paths = [CODEBASE_PATH, gem5_src_path]
    print(f"Using include paths: {include_paths}")

    all_chunks = []
    # Use tqdm to create a progress bar for the file parsing loop
    with tqdm(total=len(all_files), desc="Parsing files") as pbar:
        for file_path in all_files:
            # Pass the include paths to the parsing function
            chunks = parse_file(file_path, include_paths)
            all_chunks.extend(chunks)
            pbar.update(1)

    if not all_chunks:
        print("❌ ERROR: No chunks were extracted from any files. Halting.")
        return
    print(f"✅ Extracted a total of {len(all_chunks)} semantic chunks.")

    print("\n--- Step 3/5: Initializing Vector Database ---")
    client = chromadb.PersistentClient(path=DB_PATH)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Deleting existing collection '{COLLECTION_NAME}' for a fresh start.")
        client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"✅ ChromaDB collection '{COLLECTION_NAME}' is ready.")

    print("\n--- Step 4/5: Loading Embedding Model ---")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Embedding model loaded successfully.")

    print("\n--- Step 5/5: Generating Embeddings and Storing in DB (in batches) ---")
    batch_size = 256  # Process 256 chunks at a time
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches", total=total_batches):
        batch = all_chunks[i:i+batch_size]
        
        documents = [item['text'] for item in batch]
        metadatas = [item['metadata'] for item in batch]
        # Create unique IDs for each chunk
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        
        embeddings = embedding_model.encode(documents).tolist()
        
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"\n🎉 Ingestion complete! Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    run_ingestion()
