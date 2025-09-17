import os
import json
import clang.cindex
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import subprocess

# Import configuration
from config import CODEBASE_PATH, DB_PATH, EMBEDDING_MODEL_NAME, COLLECTION_NAME

# --- CRITICAL FIX: EXPLICITLY SET LIBCLANG PATH ---
# Find this path by running: find /usr -name "libclang.so*"
# Example path: '/usr/lib/x86_64-linux-gnu/libclang.so.1'
LIBCLANG_PATH = '/usr/lib/llvm-18/lib/libclang.so.1' # <-- UPDATE WITH YOUR PATH
if os.path.exists(LIBCLANG_PATH):
    clang.cindex.Config.set_library_file(LIBCLANG_PATH)
    print(f"✅ Set libclang path to: {LIBCLANG_PATH}")
else:
    print(f"⚠️ WARNING: libclang path not found at '{LIBCLANG_PATH}'. Parsing may fail.")
    print("Please find your libclang.so file and update the LIBCLANG_PATH variable.")

# Node kinds to extract from the AST
NODE_KINDS_TO_EXTRACT = [
    clang.cindex.CursorKind.FUNCTION_DECL,
    clang.cindex.CursorKind.CXX_METHOD,
    clang.cindex.CursorKind.CLASS_DECL,
    clang.cindex.CursorKind.STRUCT_DECL,
]

def get_system_include_paths():
    """
    Retrieves the default system include paths from the compiler.
    This is the key to solving 'stddef.h' not found and similar errors.
    """
    try:
        # Use clang++ as it's what the build was configured for
        process = subprocess.run(
            ['clang++', '-E', '-Wp,-v', '-'],
            input='',
            capture_output=True,
            text=True,
            check=True
        )
        # Search for the include path list in the stderr
        lines = process.stderr.split('\n')
        start_index = -1
        end_index = -1
        for i, line in enumerate(lines):
            if '#include <...> search starts here:' in line:
                start_index = i + 1
            elif 'End of search list.' in line:
                end_index = i
                break
        
        if start_index != -1 and end_index != -1:
            paths = [f"-I{path.strip()}" for path in lines[start_index:end_index]]
            print(f"✅ Found {len(paths)} system include paths.")
            return paths
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ WARNING: Could not determine system include paths. Parsing may have errors.")
        return []
    return []

def load_compilation_database(codebase_path: str) -> list:
    """Loads the compile_commands.json file."""
    compdb_path = os.path.join(codebase_path, 'compile_commands.json')
    if not os.path.exists(compdb_path):
        print(f"❌ ERROR: compile_commands.json not found at '{compdb_path}'")
        return None
    
    with open(compdb_path, 'r') as f:
        compilation_data = json.load(f)
    
    print(f"✅ Loaded compilation database with {len(compilation_data)} entries.")
    return compilation_data

def extract_chunks_from_ast(cursor: clang.cindex.Cursor, file_content: str, target_filename: str) -> list[dict]:
    """Recursively traverses the AST and extracts semantically complete chunks."""
    chunks = []
    try:
        if cursor.location.file and os.path.samefile(cursor.location.file.name, target_filename):
            if cursor.kind in NODE_KINDS_TO_EXTRACT:
                start = cursor.extent.start.offset
                end = cursor.extent.end.offset
                code_text = file_content[start:end]

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
    except (FileNotFoundError, ValueError):
        pass
    return chunks

def run_ingestion():
    """Main function to run the entire high-fidelity ingestion pipeline."""
    
    print("--- Step 1/3: Loading Compilation Database & System Paths ---")
    comp_db = load_compilation_database(CODEBASE_PATH)
    if not comp_db:
        return
    system_includes = get_system_include_paths()

    print("\n--- Step 2/3: Parsing All Files (Single-Threaded) ---")
    all_chunks = []
    error_count = 0

    for entry in tqdm(comp_db, desc="Parsing files"):
        try:
            # Construct absolute path for the file to be parsed
            file_path = os.path.join(entry['directory'], entry['file'])
            
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()

            # Sanitize arguments from the compilation database
            args = entry.get('arguments', [])
            sanitized_args = []
            skip_next = False
            for i, arg in enumerate(args):
                if skip_next or i == 0: # Skip compiler and handle flags
                    skip_next = False
                    continue
                if arg == '-o':
                    skip_next = True
                    continue
                if arg == entry['file'] or arg == file_path:
                    continue
                sanitized_args.append(arg)

            # Combine sanitized args with system includes
            final_args = sanitized_args + system_includes

            index = clang.cindex.Index.create()
            tu = index.parse(file_path, args=final_args, 
                             options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            
            fatal_errors = [d for d in tu.diagnostics if d.severity >= clang.cindex.Diagnostic.Error]
            if fatal_errors:
                error_count += 1
                continue
                
            chunks = extract_chunks_from_ast(tu.cursor, file_content, file_path)
            all_chunks.extend(chunks)
        except Exception:
            error_count += 1

    if error_count > 0:
        print(f"\n--- ❗ NOTE: Encountered {error_count} files with non-fatal or recoverable parsing errors. ---")

    if not all_chunks:
        print("\n❌ ERROR: No chunks were extracted. All files may have had fatal parsing errors.")
        return
        
    print(f"\n✅ Extracted a total of {len(all_chunks)} semantic chunks.")

    print("\n--- Step 3/3: Embedding and Storing Chunks ---")
    client = chromadb.PersistentClient(path=DB_PATH)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Deleting existing collection '{COLLECTION_NAME}'.")
        client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    batch_size = 256
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches"):
        batch = all_chunks[i:i+batch_size]
        documents = [item['text'] for item in batch]
        metadatas = [item['metadata'] for item in batch]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        embeddings = embedding_model.encode(documents).tolist()
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"\n🎉 Ingestion complete! Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    run_ingestion()

