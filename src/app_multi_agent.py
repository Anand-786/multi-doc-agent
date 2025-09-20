# app.py - REFACTORED FOR NEW UI FLOW

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import streamlit as st
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- Configuration & Templates ---
load_dotenv()
DB_PATH = "multi_agent_chroma_db"
PROMPT_TEMPLATE = """
You are a helpful and precise expert on the document provided.
A user has asked the following question: "{user_query}"
Here are the most relevant sections from the document:
---CONTEXT---
{context_string}
---END CONTEXT---
Your task is to formulate a clear and concise answer to the user's question, using ONLY the information found in the context blocks.
If the information in the context is not sufficient, state: "I could not find a definitive answer in the provided document."
"""

# --- Resource Loading ---
@st.cache_resource
def load_resources():
    """Loads all the necessary models and database clients."""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel('gemini-1.5-flash')
    except KeyError:
        st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
        return None, None, None
    return embedding_model, client, llm

embedding_model, db_client, llm = load_resources()

# --- Agent Creation & Deletion Logic ---
def process_and_store_pdf(pdf_file, collection_name, client):
    """Processes an uploaded PDF and creates a new agent (ChromaDB collection)."""
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if not any(page.get_text() for page in doc):
            st.error("Could not extract text from PDF. The file may be empty or image-based.")
            return False
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        pages_content = [page.get_text() for page in doc]
        metadatas = [{"page_number": i+1} for i in range(len(doc))]
        chunks = text_splitter.create_documents(pages_content, metadatas=metadatas)
        
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_metadatas = [chunk.metadata for chunk in chunks]

        collection = client.create_collection(name=collection_name)
        collection.add(
            ids=[f"{collection_name}_{i}" for i in range(len(chunk_texts))],
            documents=chunk_texts,
            metadatas=chunk_metadatas
        )
        return True
    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
        return False

# --- UI Sections ---
def agent_selection_ui():
    """UI for the main agent selection screen."""
    st.subheader("Select an Agent")

    # Using columns for a card-like layout
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("##### ✦ Gem5 Doc Expert")
            st.markdown("Based on official gem5 docs for computer architecture simulation.")
            if st.button("Select ➜", use_container_width=True):
                st.session_state.active_agent = "gem5_expert"
                st.session_state.page = "chat"
                st.rerun()
    
    st.divider()
    
    st.subheader("Or, create your own")
    uploaded_file = st.file_uploader("Upload a PDF to create a new document agent", type="pdf")

    if uploaded_file:
        new_agent_name = uploaded_file.name.split('.pdf')[0].replace(" ", "_")
        if st.button(f"Create '{new_agent_name}' Agent", use_container_width=True):
            with st.spinner(f"Creating agent from {uploaded_file.name}..."):
                collection_name = "".join(e for e in new_agent_name if e.isalnum() or e in ('_', '-'))
                
                # Check if collection already exists
                if collection_name in [c.name for c in db_client.list_collections()]:
                     st.warning(f"An agent with a similar name ('{collection_name}') already exists. Please upload a file with a different name.")
                else:
                    success = process_and_store_pdf(uploaded_file, collection_name, db_client)
                    if success:
                        st.session_state.agent_list[new_agent_name] = collection_name
                        st.session_state.active_agent = new_agent_name
                        st.session_state.page = "chat" # Switch to chat page
                        st.success("Agent created successfully!")
                        st.rerun()

def chat_ui():
    """UI for the chat interface."""
    active_agent_name = st.session_state.active_agent
    st.info(f"Active: **{active_agent_name}**")

    if active_agent_name not in st.session_state.messages:
        st.session_state.messages[active_agent_name] = []

    for message in st.session_state.messages[active_agent_name]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask {active_agent_name} a question..."):
        if not llm:
            st.error("LLM not initialized. Check API Key.")
            return

        st.session_state.messages[active_agent_name].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            try:
                collection_name = st.session_state.agent_list[active_agent_name]
                collection = db_client.get_collection(name=collection_name)
                
                results = collection.query(query_texts=[prompt], n_results=5)
                context_string = "\n\n".join(results['documents'][0])
                
                final_prompt = PROMPT_TEMPLATE.format(user_query=prompt, context_string=context_string)
                response = llm.generate_content(final_prompt)
                
                st.session_state.messages[active_agent_name].append({"role": "assistant", "content": response.text})
                with st.chat_message("assistant"):
                    st.markdown(response.text)
            except Exception as e:
                st.error(f"An error occurred while getting the response: {e}")


# --- Main Application Logic ---
st.set_page_config(page_title="Multi-Agent Document Platform", layout="centered")
st.title(" ◆ Multi-Agent Document Platform")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "agent_list" not in st.session_state:
    st.session_state.agent_list = {"gem5_expert": "gem5_documentation_v3"} 
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
if "page" not in st.session_state:
    st.session_state.page = "selection" # 'selection' or 'chat'

# --- Sidebar for Agent Navigation & Management ---
with st.sidebar:
    # Button to go back to the selection screen, only shown during chat
    if st.session_state.page == "chat":
        if st.button("⬅ Back to Agent Selection", use_container_width=True):
            st.session_state.page = "selection"
            st.session_state.active_agent = None
            st.rerun()

    st.divider()

    # List of available agents for quick switching
    st.header("Available Agents")
    for agent_name in list(st.session_state.agent_list.keys()):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            # Highlight the active agent
            button_type = "primary" if agent_name == st.session_state.active_agent else "secondary"
            if st.button(agent_name, use_container_width=True, key=f"select_{agent_name}", type=button_type):
                st.session_state.active_agent = agent_name
                st.session_state.page = "chat"
                st.toast(f"Switched to agent: {agent_name}")
                st.rerun()
        with col2:
            if agent_name != "gem5_expert":
                if st.button("🗑️", key=f"delete_{agent_name}", help=f"Delete agent '{agent_name}'"):
                    try:
                        collection_to_delete = st.session_state.agent_list[agent_name]
                        db_client.delete_collection(name=collection_to_delete)
                        del st.session_state.agent_list[agent_name]
                        if agent_name in st.session_state.messages:
                            del st.session_state.messages[agent_name]
                        
                        st.toast(f"Agent '{agent_name}' deleted!")
                        # If the deleted agent was active, go back to selection screen
                        if st.session_state.active_agent == agent_name:
                            st.session_state.page = "selection"
                            st.session_state.active_agent = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting agent: {e}")

# --- Page Routing ---
if st.session_state.page == "selection":
    agent_selection_ui()
else:
    chat_ui()