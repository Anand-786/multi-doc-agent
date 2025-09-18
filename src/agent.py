import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

DB_PATH = "gem5_chroma_db"
COLLECTION_NAME = "gem5_documentation"
MODEL_NAME = "all-MiniLM-L6-v2"

try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("Error: GOOGLE_API_KEY not found in environment variables or .env file.")
    exit()

print("Initializing models and database connection...")
embedding_model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
llm = genai.GenerativeModel('gemini-1.5-flash-latest') 
print("Initialization complete.")

def main():
    """
    Main loop to handle user queries.
    """
    print("\n-gem5 Documentation Q&A Agent-")
    print("Ask a question about gem5, or type 'exit' to quit.")

    while True:
        user_query = input("\n> ")
        if user_query.lower() == 'exit':
            break

        query_embedding = embedding_model.encode(user_query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=['documents', 'metadatas']
        )
        
        retrieved_docs = results['documents'][0]
        retrieved_metadatas = results['metadatas'][0]
        
        context_string = ""
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadatas)):
            context_string += f"Source {i+1}:\n"
            context_string += f"  - Page Title: {meta.get('page_title', 'N/A')}\n"
            context_string += f"  - Section: {meta.get('section_heading', 'N/A')}\n"
            context_string += f"  - URL: {meta.get('source_url', 'N/A')}\n"
            context_string += f"  - Content: {doc}\n\n"

        prompt = f"""
        You are an expert assistant for the gem5 computer architecture simulator.
        A user has asked the following question:
        "{user_query}"

        Here is the most relevant context from the gem5 documentation to help you answer the question:
        ---CONTEXT---
        {context_string}
        ---END CONTEXT---

        Your task is to provide a clear and concise answer to the user's question based ONLY on the provided context. After your answer, you MUST list the sources you used. For each source, provide the "Page Title" and the "URL". Format them as a list under a "Sources:" heading.

        If the context is not sufficient to answer the question, you must state that you cannot answer based on the provided documentation.
        """

        print("\nThinking...")
        try:
            response = llm.generate_content(prompt)
            print("\nAnswer:\n" + response.text)
        except Exception as e:
            print(f"An error occurred while generating the response: {e}")


if __name__ == "__main__":
    main()