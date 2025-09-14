# src/agent.py
import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Import configuration from our config file
from config import DB_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, COLLECTION_NAME, ENV_PATH

def main():
    """
    Main function to initialize models and start the chat loop.
    """
    # Load API key from .env file
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    genai.configure(api_key=api_key)

    # --- Initialize Models and DB Connection ---
    print("🔍 Initializing models and database connection...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    llm = genai.GenerativeModel(LLM_MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    print("✅ Ready to chat!")

    # --- Chat Loop ---
    while True:
        user_question = input("\nAsk a question about the codebase (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break

        print("\n🤖 Thinking...")

        # 1. Embed the user's query
        query_embedding = embedding_model.encode(user_question).tolist()

        # 2. Search the database for relevant context
        context_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5 # Retrieve 5 chunks for context
        )
        context = "\n---\n".join(context_results['documents'][0])

        # 3. Build the prompt
        prompt = f"""
        You are an expert on a C++ codebase for a Tomasulo simulator.
        Answer the user's question based ONLY on the following code context.
        If the answer is not in the context, state that clearly.

        CONTEXT:
        {context}

        QUESTION:
        {user_question}

        ANSWER:
        """

        # 4. Call the LLM to generate the answer
        response = llm.generate_content(prompt)
        print("\nAgent Response:")
        print(response.text)
        print("-" * 20)

if __name__ == "__main__":
    main()