import streamlit as st
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- Load environment variables from .env file ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Gem5 Docs Agent",
    page_icon="logo.png",
    layout="centered"
)

# --- Caching Models and DB Connection ---
# Use st.cache_resource to load models and DB only once
@st.cache_resource
def load_resources():
    print("Initializing resources...")
    # Initialize the embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize the ChromaDB client
    client = chromadb.PersistentClient(path="gem5_chroma_db_v3")
    collection = client.get_collection(name="gem5_documentation_v3")
    
    # Configure and initialize the Gemini model
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    except KeyError:
        st.error("🚨 GOOGLE_API_KEY not found. Please set it in your .env file.")
        return None, None, None
        
    print("✅ Resources initialized.")
    return embedding_model, collection, llm

embedding_model, collection, llm = load_resources()

# --- Main App Logic ---
st.title("Gem5 Documentation Agent")
st.caption("Ask any question about the gem5 simulator documentation.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the Ruby memory system?"):
    if not llm:
        st.error("Generative model not initialized due to missing API key.")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # --- Agent's Thinking Process ---
        with st.spinner("Thinking..."):
            # 1. Embed the user's query
            query_embedding = embedding_model.encode(prompt).tolist()

            # 2. Query the database
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=['documents', 'metadatas']
            )
            
            retrieved_docs = results['documents'][0]
            retrieved_metadatas = results['metadatas'][0]
            
            # 3. Construct the prompt for the LLM
            context_string = ""
            sources_string = ""
            source_links = set() # Use a set to store unique links

            for i, meta in enumerate(retrieved_metadatas):
                context_string += f"--- Context {i+1} ---\n{retrieved_docs[i]}\n\n"
                # Add unique source links to the set
                source_links.add(f"[{meta.get('page_title', 'Unknown Title')}]({meta.get('source_url', '#')})")
            
            # Create a clean, bulleted list of sources
            sources_string = "\n".join(f"- {link}" for link in source_links)

            final_prompt = f"""
            You are an expert assistant for the gem5 computer architecture simulator.
            A user has asked the following question: "{prompt}"

            Here is the most relevant context from the documentation:
            ---CONTEXT---
            {context_string}
            ---END CONTEXT---

            Your task is to provide a clear and concise answer to the user's question based ONLY on the provided context.
            After your answer, you MUST provide a detailed list of the sources you used. Format them under a "Sources:" heading.
            For each source, list it on a new line. Include the 'Page Title' and the 'Section Heading'. After the text, include the full URL in square brackets.
            
            Example Format:
            - From: Page Title - Section Heading [https://www.gem5.org/...]

            If the context is insufficient, state that you cannot answer based on the provided documentation.
            """

            # 4. Generate and display the response
            response = llm.generate_content(final_prompt)
            answer = response.text
            
            # Combine the answer and sources for display
            final_response = f"{answer}\n\n---\n**Sources:**\n{sources_string}"

            with st.chat_message("assistant"):
                st.markdown(final_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": final_response})