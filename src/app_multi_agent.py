# app.py - REFACTORED FOR MULTI-AGENT SUPPORT

# New import for reading PDFs
import fitz  # PyMuPDF
# New import for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter 

import streamlit as st
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- Configuration & Templates ---
load_dotenv()
DB_PATH = "multi_agent_chroma_db" # New path for all our agent DBs
PROMPT_TEMPLATE = """
You are a helpful and precise expert on the document provided.

A user has asked the following question:
"{user_query}"

Here are the most relevant sections from the document. Each block includes the source page number.
---CONTEXT---
{context_string}
---END CONTEXT---

Your task is to formulate a clear and concise answer to the user's question, using ONLY the information found in the context blocks.

If the information in the context is not sufficient to answer the question, simply state: "I could not find a definitive answer in the provided document."
"""

# --- Resource Loading ---
@st.cache_resource
def load_resources():
    """Loads all the necessary models and database clients."""
    print("Initializing resources...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # We will use a persistent client that can access multiple collections
    client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel('gemini-1.5-flash') # Updated model for better performance
    except KeyError:
        st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
        return None, None, None
        
    print("Resources initialized.")
    return embedding_model, client, llm

embedding_model, db_client, llm = load_resources()

# --- NEW: Agent Creation Logic ---
def process_and_store_pdf(pdf_file, collection_name, client, emb_model):
    """Processes an uploaded PDF and creates a new agent (ChromaDB collection)."""
    # 1. Extract Text
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "\n".join([page.get_text() for page in doc])
    metadatas = [{"page_number": i+1} for i in range(len(doc))]
    doc.close()

    if not full_text.strip():
        st.error("Could not extract text from PDF.")
        return False
        
    # 2. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Split documents by page to keep metadata aligned
    pages_content = [page.get_text() for page in fitz.open(stream=pdf_bytes, filetype="pdf")]
    chunks = text_splitter.create_documents(pages_content, metadatas=metadatas)
    
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_metadatas = [chunk.metadata for chunk in chunks]

    # 3. Create a new collection and add the data
    collection = client.create_collection(name=collection_name)
    collection.add(
        ids=[f"{collection_name}_{i}" for i in range(len(chunk_texts))],
        documents=chunk_texts,
        metadatas=chunk_metadatas
    )
    return True

# --- UI & Application Logic ---
st.set_page_config(page_title="Multi-Agent Chat", layout="wide")
st.title("🤖 Multi-Agent Chat")
st.caption("Create custom agents from your PDFs or chat with pre-built ones.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "agent_list" not in st.session_state:
    # We start with the gem5 agent as a pre-built option.
    # Make sure your original gem5 DB is copied to the new DB_PATH folder.
    st.session_state.agent_list = {"gem5_expert": "gem5_documentation_v3"} 
if "active_agent" not in st.session_state:
    st.session_state.active_agent = "gem5_expert"

# --- Sidebar for Agent Management ---
with st.sidebar:
    st.header("Agents")

    # Display buttons for each agent
    for agent_name in st.session_state.agent_list.keys():
        if st.button(agent_name, use_container_width=True):
            st.session_state.active_agent = agent_name
            st.toast(f"Switched to agent: {agent_name}")

    st.divider()
    st.header("Create New Agent")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        new_agent_name = uploaded_file.name.split('.pdf')[0].replace(" ", "_")
        if st.button(f"Create '{new_agent_name}' Agent"):
            with st.spinner(f"Creating agent from {uploaded_file.name}..."):
                # Use a sanitized name for the collection
                collection_name = "".join(e for e in new_agent_name if e.isalnum() or e in ('_', '-'))
                
                success = process_and_store_pdf(uploaded_file, collection_name, db_client, embedding_model)
                if success:
                    st.session_state.agent_list[new_agent_name] = collection_name
                    st.session_state.active_agent = new_agent_name
                    st.success("Agent created successfully!")
                    st.rerun() # Refresh the sidebar to show the new agent button
                else:
                    st.error("Failed to create agent.")

# --- Main Chat Interface ---
active_agent_name = st.session_state.active_agent
st.info(f"Currently chatting with: **{active_agent_name}**")

# Ensure message history for the active agent is initialized
if active_agent_name not in st.session_state.messages:
    st.session_state.messages[active_agent_name] = []

# Display previous messages for the active agent
for message in st.session_state.messages[active_agent_name]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input(f"Ask {active_agent_name} a question..."):
    if not llm:
        st.error("LLM not initialized.")
    else:
        # Add user message to history and display it
        st.session_state.messages[active_agent_name].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            # Get the correct collection name for the active agent
            collection_name = st.session_state.agent_list[active_agent_name]
            collection = db_client.get_collection(name=collection_name)
            
            # Query the database
            results = collection.query(query_texts=[prompt], n_results=5)

            # Build the context
            retrieved_docs = results['documents'][0]
            context_string = "\n\n".join(retrieved_docs)
            
            # Format the final prompt and get response
            final_prompt = PROMPT_TEMPLATE.format(user_query=prompt, context_string=context_string)
            response = llm.generate_content(final_prompt)
            final_response = response.text

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(final_response)
            
            # Add assistant response to history
            st.session_state.messages[active_agent_name].append({"role": "assistant", "content": final_response})