import json
import chromadb
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers.SentenceTransformer")

CHUNKS_FILE = "gem5_docs_chunks2.json"
DB_PATH = "gem5_chroma_db_v3"
COLLECTION_NAME = "gem5_documentation_v3"
MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    print(f"Loading processed chunks from {CHUNKS_FILE}...")
    try:
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {CHUNKS_FILE} was not found.")
        return

    print(f"Initializing sentence-transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.")

    print(f"Initializing ChromaDB client at: {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    print(f"Getting or creating ChromaDB collection: {COLLECTION_NAME}...")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print("Collection ready.")

    print("Preparing data for embedding and storage...")
    
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    documents = [chunk['text'] for chunk in all_chunks]
    metadatas = [chunk['metadata'] for chunk in all_chunks]
    
    print(f"Generating embeddings for {len(documents)} chunks...")
    
    embeddings = model.encode(documents, show_progress_bar=True)
    
    print("Adding data to the collection...")
    
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end_index = i + batch_size
        print(f"  - Adding batch {i//batch_size + 1}...")
        
        collection.add(
            ids=ids[i:end_index],
            embeddings=embeddings[i:end_index].tolist(),
            documents=documents[i:end_index],
            metadatas=metadatas[i:end_index]
        )
    print("\nSuccess! All chunks have been embedded and stored in ChromaDB.")
    print(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    main()