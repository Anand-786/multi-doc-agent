import os
import clang.cindex
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIGURATION ---
TEST_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db_test'))
TEST_COLLECTION_NAME = "cpp_test_collection"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TEST_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'complex_test.cc'))

NODE_KINDS_TO_EXTRACT = [
    clang.cindex.CursorKind.FUNCTION_DECL,
    clang.cindex.CursorKind.CXX_METHOD,
    clang.cindex.CursorKind.CLASS_DECL,
    clang.cindex.CursorKind.STRUCT_DECL,
]

# --- MODIFIED FUNCTION WITH DEBUGGING ---
def extract_chunks_from_ast(cursor: clang.cindex.Cursor, file_content: str, target_filename: str) -> list[dict]:
    """
    Recursively traverses the AST and extracts semantically complete chunks,
    with added print statements for debugging the file path comparison.
    """
    chunks = []
    
    # We only care about nodes that have a file location.
    if not cursor.location.file:
        # Recurse into children even if the parent has no file, as they might.
        # This handles the TranslationUnit, which sometimes has a null file.
        for child in cursor.get_children():
            chunks.extend(extract_chunks_from_ast(child, file_content, target_filename))
        return chunks

    # --- DEBUGGING PRINTS ---
    # This block will now only run for cursors that have a file path.
    # We'll print the comparison for the first few nodes from the target file.
    if cursor.spelling: # Only print for named nodes to reduce noise
        is_same = False
        try:
            is_same = os.path.samefile(cursor.location.file.name, target_filename)
        except FileNotFoundError:
            pass # Ignore files that might not exist, like compiler built-ins
            
        # Print a diagnostic line
        print(f"[DEBUG] Node: '{cursor.spelling}' ({cursor.kind.name}) | In File: '{cursor.location.file.name}' | Matches Target: {is_same}")

    # The actual filtering logic
    try:
        if os.path.samefile(cursor.location.file.name, target_filename):
            
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
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during traversal: {e}")

    return chunks

def run_test_ingestion():
    """
    Main function to run the test ingestion pipeline on a single file.
    """
    print(f"--- 1/4: Reading Test File ---")
    if not os.path.exists(TEST_FILE_PATH):
        print(f"ERROR: Test file not found at {TEST_FILE_PATH}")
        return
    
    with open(TEST_FILE_PATH, 'r', encoding='utf-8') as f:
        file_content = f.read()
    print(f"✅ Read {os.path.basename(TEST_FILE_PATH)}")

    print(f"\n--- 2/4: Parsing C++ and Extracting Chunks ---")
    index = clang.cindex.Index.create()
    
    print(f"Target file for parsing is: {TEST_FILE_PATH}") # Add a print for our target
    
    translation_unit = index.parse(TEST_FILE_PATH, args=['-std=c++11'])
    
    if not translation_unit:
        print("❌ Failed to create translation unit. Check libclang installation.")
        return
    
    parsing_errors = [d for d in translation_unit.diagnostics if d.severity >= d.Error]
    if parsing_errors:
        print("❌ Encountered parsing errors:")
        for diag in parsing_errors:
            print(f"  - {diag}")
        
    chunks = extract_chunks_from_ast(translation_unit.cursor, file_content, TEST_FILE_PATH)
    
    if not chunks:
        print("❌ No chunks were extracted. Check the file content and NODE_KINDS_TO_EXTRACT.")
        return
        
    print(f"✅ Extracted {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  - Type: {chunk['metadata']['type']}, Name: '{chunk['metadata']['name']}'")
        
    print(f"\n--- 3/4: Setting up Test Vector Database ---")
    client = chromadb.PersistentClient(path=TEST_DB_PATH)
    
    if TEST_COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Deleting existing collection: {TEST_COLLECTION_NAME}")
        client.delete_collection(name=TEST_COLLECTION_NAME)
        
    collection = client.create_collection(name=TEST_COLLECTION_NAME)
    print("✅ Test database and collection are ready.")
    
    print(f"\n--- 4/4: Embedding Chunks and Storing in DB ---")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    documents = [item['text'] for item in chunks]
    metadatas = [item['metadata'] for item in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    print("Generating embeddings...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()
    
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    
    print(f"\n🎉 Ingestion complete! Added {collection.count()} documents to the test database.")

if __name__ == "__main__":
    run_test_ingestion()

