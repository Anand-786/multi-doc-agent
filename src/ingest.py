# src/ingest.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
import tqdm

# Import configuration from our config file
from config import CODEBASE_PATH, DB_PATH, EMBEDDING_MODEL_NAME, COLLECTION_NAME, CHUNK_SIZE_LINES, OVERLAP_LINES

def parse_and_chunk_codebase(directory, chunk_size, overlap):
    """
    Parses files by splitting them into fixed-size, overlapping chunks of lines.
    """
    all_chunks = []
    print(f"\n[INFO] Starting to scan and process files in: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".cpp", ".h")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    if not lines: continue

                    start_line = 0
                    while start_line < len(lines):
                        end_line = start_line + chunk_size
                        chunk_lines = lines[start_line:end_line]
                        chunk_text = "".join(chunk_lines)
                        all_chunks.append({
                            "text": chunk_text,
                            "metadata": {"filename": file_path, "start_line": start_line + 1}
                        })
                        next_start = start_line + chunk_size - overlap
                        start_line = next_start if next_start > start_line else start_line + 1
                except Exception as e:
                    print(f"  -> ERROR processing {file_path}: {e}")
    print(f"[INFO] Finished parsing. Found {len(all_chunks)} total code chunks.")
    return all_chunks

def run_ingestion():
    """
    Main function to run the entire ingestion pipeline.
    """
    print("--- Step 1/4: Parsing codebase ---")
    chunks = parse_and_chunk_codebase(CODEBASE_PATH, CHUNK_SIZE_LINES, OVERLAP_LINES)
    if not chunks:
        print("No chunks found. Exiting.")
        return

    print("\n--- Step 2/4: Loading embedding model ---")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Model loaded successfully.")

    print("\n--- Step 3/4: Initializing Vector Database ---")
    client = chromadb.PersistentClient(path=DB_PATH)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Deleting existing collection '{COLLECTION_NAME}' to ensure a fresh start.")
        client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)
    print("✅ ChromaDB initialized and collection is ready.")

    print("\n--- Step 4/4: Generating Embeddings and Storing in DB ---")
    batch_size = 50
    for i in tqdm.tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i+batch_size]
        documents = [item['text'] for item in batch]
        metadatas = [item['metadata'] for item in batch]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        embeddings = embedding_model.encode(documents).tolist()
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"\n🎉 Ingestion complete! Total documents: {collection.count()}")

if __name__ == "__main__":
    # This block allows the script to be run directly from the terminal
    run_ingestion()