import streamlit as st
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

PROMPT_TEMPLATE = """
You are a helpful and precise expert on the gem5 computer architecture simulator.

A user has asked the following question:
"{user_query}"

Here are the most relevant sections from the gem5 documentation. Each block includes the source URL, page title, and heading.

---CONTEXT---
{context_string}
---END CONTEXT---

Your tasks are:
1.  Carefully read the context provided.
2.  Formulate a clear and concise answer to the user's question, using **ONLY** the information found in the context blocks. Do not use any prior knowledge.
3.  After the answer, create a markdown heading titled "Sources".
4.  Under the "Sources" heading, create a bulleted list of the documents you used. For each source, you **MUST** format it exactly as follows, using the metadata from the context block:
    * **From:** Page Title - Full Heading [Source URL]

If the information in the context is not sufficient to answer the question, simply state: "I could not find a definitive answer in the provided documentation."
"""

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
            # 1. Query the database using the user's prompt directly
            results = collection.query(
                query_texts=[prompt],
                n_results=5, # Using 5 results gives the model more context
                include=['documents', 'metadatas']
            )

            retrieved_docs = results.get('documents', [[]])[0]
            retrieved_metadatas = results.get('metadatas', [[]])[0]

            # Handle case where no documents are found
            if not retrieved_docs:
                final_response = "I'm sorry, I couldn't find any relevant information in the documentation to answer your question."
            else:
                # 2. CRITICAL: Build the context string WITH all the source metadata
                context_blocks = []
                for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metadatas)):
                    page_title = meta.get('page_title', 'N/A')
                    source_url = meta.get('source_url', 'N/A')
                    # This assumes your chunker provides these keys
                    parent_section = meta.get('parent_section', '') 
                    section_heading = meta.get('section_heading', '')
                    full_heading = f"{parent_section} - {section_heading}".strip(" -")
                    
                    context_block = f"""--- START OF CONTEXT BLOCK {i+1} ---
            Source URL: {source_url}
            Page Title: {page_title}
            Full Heading: {full_heading}

            Content:
            {doc}
            --- END OF CONTEXT BLOCK {i+1} ---"""
                    context_blocks.append(context_block)
                
                context_string = "\n\n".join(context_blocks)

                # 3. Format the final prompt using the new template
                final_prompt = PROMPT_TEMPLATE.format(
                    user_query=prompt,
                    context_string=context_string
                )

                # 4. Generate the response. The LLM will now create the answer AND the sources.
                response = llm.generate_content(final_prompt)
                final_response = response.text # The final response is just the model's text

            # Display the assistant's response
            with st.chat_message("assistant"):
                st.markdown(final_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": final_response})