import chromadb

# --- CONFIGURATION ---
# Point to the TEST database path and collection name
DB_PATH = "db_test/"
COLLECTION_NAME = "cpp_test_collection"

def check_database():
    """
    Connects to the database and prints its contents.
    """
    print(f"--- Attempting to connect to database at: {DB_PATH} ---")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        count = collection.count()
        if count == 0:
            print("❌ Collection is empty.")
            return

        print(f"✅ Success! Found {count} documents in the '{COLLECTION_NAME}' collection.")
        
        # Retrieve all documents to inspect them
        results = collection.get(include=["metadatas", "documents"])
        
        print("\n--- Inspecting Documents ---")
        for i in range(len(results["ids"])):
            metadata = results["metadatas"][i]
            document = results["documents"][i]
            
            print(f"\n--- Document {i+1} ---")
            print(f"  Metadata:")
            print(f"    - Type: {metadata.get('type', 'N/A')}")
            print(f"    - Name: {metadata.get('name', 'N/A')}")
            print(f"    - File: {metadata.get('file_path', 'N/A')}")
            print(f"    - Lines: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}")
            print(f"  Document Text (first 50 chars):")
            print(f"    '{document[:50].strip().replace('\n', ' ')}...'")
        print("\n--------------------------")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("   Please ensure you have run the 'ast_parser_test.py' script first.")

if __name__ == "__main__":
    check_database()
