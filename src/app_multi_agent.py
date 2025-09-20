__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import streamlit as st
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import io
import yake
import random
import time

load_dotenv()
DB_PATH = "multi_agent_chroma_db"
CLASSIFIER_PATH = "intent_classifiers"
PROMPT_TEMPLATE = """
You are a helpful and precise expert on the document provided. A user has asked the following question: "{user_query}".
Here are the most relevant sections from the document:
---CONTEXT---
{context_string}
---END CONTEXT---
Formulate a clear and concise answer using ONLY the information in the context. If the information is not sufficient, state: "I could not find a definitive answer in the provided document."
"""
GEM5_PROMPT_TEMPLATE = """
You are a helpful and precise expert on the gem5 computer architecture simulator. A user has asked the following question: "{user_query}".
Here are the most relevant sections from the gem5 documentation. Each block includes the source URL, page title, and heading.
---CONTEXT---
{context_string}
---END CONTEXT---
Your tasks are:
1.  Carefully read the context provided.
2.  Formulate a clear and concise answer to the user's question, using **ONLY** the information found in the context blocks. Do not use any prior knowledge.
3.  After the answer, create a markdown heading titled "Sources".
4.  Under the "Sources" heading, create a bulleted list of the documents you used. For each source, you **MUST** format it exactly as follows, using the metadata from the context block:
    * **From:** Page Title - [Source URL]
If the information in the context is not sufficient, state: "I could not find a definitive answer in the provided documentation."
"""

@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel('gemini-1.5-flash')
    except KeyError:
        st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
        llm = None
    if not os.path.exists(CLASSIFIER_PATH):
        os.makedirs(CLASSIFIER_PATH)
    return embedding_model, client, llm

embedding_model, db_client, llm = load_resources()

def ensure_encoder_exists(agent_name):
    encoder_path = os.path.join(CLASSIFIER_PATH, f"{agent_name}_encoder.joblib")
    if not os.path.exists(encoder_path):
        st.warning(f"{agent_name} encoder not found. Creating it now...")
        encoder = LabelEncoder()
        encoder.fit(['casual', 'doc_qna'])
        joblib.dump(encoder, encoder_path)
        st.success(f"{agent_name} encoder created successfully!")
        time.sleep(2)

def create_agent_with_intent_classifier(pdf_file, collection_name, client):
    st.toast("➤ Step 1/5: Chunking document..."); time.sleep(2)
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        pages_content = [page.get_text() for page in doc if page.get_text()]
        if not pages_content:
            st.error("Could not extract text from PDF."); return False, None
        all_chunks = [chunk for i, content in enumerate(pages_content) for chunk in text_splitter.create_documents([content], metadatas=[{"page_number": i + 1}])]
        chunk_texts = [chunk.page_content for chunk in all_chunks]
        chunk_metadatas = [chunk.metadata for chunk in all_chunks]
        collection = client.create_collection(name=collection_name)
        collection.add(ids=[f"{collection_name}_{i}" for i in range(len(chunk_texts))], documents=chunk_texts, metadatas=chunk_metadatas)
    except Exception as e:
        st.error(f"Error during chunking: {e}"); return False, None
    st.toast("✔ Chunking complete!"); time.sleep(2)

    st.toast("➤ Step 2/5: Generating training data..."); time.sleep(2)
    kw_extractor = yake.KeywordExtractor(n=2, dedupLim=0.9, top=2, features=None)
    templates = ["What is {}?", "Can you explain {}?", "Tell me more about {}."]
    doc_qna_questions = [templates[i % len(templates)].format(kw) for chunk in chunk_texts[:min(100, len(chunk_texts))] for i, (kw, _) in enumerate(kw_extractor.extract_keywords(chunk)) if kw]
    if not doc_qna_questions:
        st.warning("Could not generate questions. Agent will be created without an intent classifier.")
        return True, None
    st.toast("✔ Training data generated!"); time.sleep(2)
    
    st.toast("➤ Step 3/5: Assembling dataset..."); time.sleep(2)
    df_doc = pd.DataFrame(doc_qna_questions, columns=['query']); df_doc['intent'] = 'doc_qna'
    df_casual = pd.read_csv('casual_queries.csv')
    df_full = pd.concat([df_doc, df_casual], ignore_index=True)
    st.toast("✔ Dataset assembled!"); time.sleep(2)

    st.toast("➤ Step 4/5: Generating embeddings..."); time.sleep(1)
    X_train, _, y_train, _ = train_test_split(df_full['query'], df_full['intent'], test_size=0.1, random_state=42)
    X_embeddings = embedding_model.encode(X_train.tolist(), show_progress_bar=True)
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    
    st.toast("➤ Step 5/5: Training classifier..."); time.sleep(2)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_embeddings, y_train_encoded)
    
    classifier_path = os.path.join(CLASSIFIER_PATH, f"{collection_name}_classifier.joblib")
    encoder_path = os.path.join(CLASSIFIER_PATH, f"{collection_name}_encoder.joblib")
    joblib.dump(classifier, classifier_path)
    joblib.dump(encoder, encoder_path)
    st.toast("✔ Classifier trained and saved!"); time.sleep(2)

    agent_config = {
        "paths": {"classifier": classifier_path, "encoder": encoder_path},
        "metadata": {
            "doc_qna_count": len(df_doc),
            "casual_qna_count": len(df_casual),
            "sample_question": doc_qna_questions[0] if doc_qna_questions else None
        }
    }
    return True, agent_config

def agent_selection_ui():
    st.subheader("Select an Agent")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("##### ✦ Gem5 Doc Agent")
            st.markdown("Based on official gem5 docs for computer architecture simulation.")
            if st.button("SELECT ▸", use_container_width=True):
                st.session_state.active_agent = "gem5_expert"
                st.session_state.page = "chat"
                st.rerun()
    st.divider()
    st.subheader("Or, create your own")
    uploaded_file = st.file_uploader("Upload a PDF to create a new document agent", type="pdf")
    if uploaded_file:
        new_agent_name = uploaded_file.name.split('.pdf')[0].replace(" ", "_")
        if st.button(f"Create '{new_agent_name}' Agent", use_container_width=True):
            collection_name = "".join(e for e in new_agent_name if e.isalnum() or e in ('_', '-'))
            if collection_name in [c.name for c in db_client.list_collections()]:
                 st.warning(f"An agent with a similar name ('{collection_name}') already exists.")
            else:
                success, agent_config = create_agent_with_intent_classifier(uploaded_file, collection_name, db_client)
                if success:
                    st.session_state.agent_list[new_agent_name] = {"collection_name": collection_name, **agent_config}
                    st.session_state.active_agent = new_agent_name
                    st.session_state.page = "chat"
                    st.success("Agent created successfully! Loading chat...")
                    st.rerun()

def display_agent_metadata(agent_name, agent_info):
    with st.container(border=True):
        st.markdown(f"#### Agent: `{agent_name}`")
        col1, col2, col3, col4 = st.columns(4)
        try:
            collection = db_client.get_collection(name=agent_info["collection_name"])
            doc_count = collection.count()
            col1.markdown(f"**Document Chunks**")
            col1.markdown(f"**`{doc_count}`**")
        except Exception:
            col1.markdown(f"**Document Chunks**")
            col1.markdown(f"**N/A**")
        
        col2.markdown("**Gen-AI Model**")
        col2.markdown(f"**`gemini-1.5-flash`**")
        
        if agent_info.get("paths"):
            col3.markdown("**Intent Classifier**")
            col3.markdown(f"**`LogisticRegression`**")
            if agent_name != "gem5_expert":
                meta = agent_info.get("metadata", {})
                col4.markdown(f"**Training Data**")
                col4.markdown(f"**`{meta.get('doc_qna_count', 'N/A')}`** : Doc | **`{meta.get('casual_qna_count', 'N/A')}`** : Casual")
        else:
            col3.metric("Intent Classifier", "None")


def chat_ui():
    active_agent_name = st.session_state.active_agent
    agent_info = st.session_state.agent_list[active_agent_name]

    display_agent_metadata(active_agent_name, agent_info)

    if active_agent_name not in st.session_state.messages:
        st.session_state.messages[active_agent_name] = []
    for message in st.session_state.messages[active_agent_name]:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    is_chat_empty = not st.session_state.messages[active_agent_name]
    sample_question = agent_info.get("metadata", {}).get("sample_question")

    if is_chat_empty and sample_question:
        st.info(f"**Sample Question :** {sample_question}")

    if prompt := st.chat_input(f"Ask {active_agent_name} a question..."):
        if not llm:
            st.error("LLM not initialized. Check API Key."); return

        st.session_state.messages[active_agent_name].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.spinner("Classifying intent..."):
            intent = "doc_qna"
            if agent_info.get("paths"):
                try:
                    classifier = joblib.load(agent_info["paths"]["classifier"])
                    encoder = joblib.load(agent_info["paths"]["encoder"])
                    query_embedding = embedding_model.encode([prompt])
                    prediction_encoded = classifier.predict(query_embedding)
                    intent = encoder.inverse_transform(prediction_encoded)[0]
                    st.toast(f"◆ Intent: {intent.replace('_', ' ').title()}")
                except FileNotFoundError: st.toast("Classifier files not found.")
                except Exception as e: st.error(f"Error during intent classification: {e}")
            else: st.toast("No classifier for this agent.")
        
        if intent == "casual":
            response = "I am a document-focused agent and can only answer questions about the document you provided."
            st.session_state.messages[active_agent_name].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"): st.markdown(response)
        else: 
            with st.spinner("Searching database..."):
                try:
                    collection = db_client.get_collection(name=agent_info["collection_name"])
                    results = collection.query(query_texts=[prompt], n_results=5, include=['metadatas', 'documents'])
                    
                    if active_agent_name == 'gem5_expert':
                        context_parts = [f"Source URL: {meta.get('source_url', 'N/A')}\nPage Title: {meta.get('page_title', 'N/A')}\nFull Heading: {meta.get('full_heading', 'N/A')}\nContent: {doc}" for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
                        context_string = "\n\n---\n\n".join(context_parts)
                        final_prompt = GEM5_PROMPT_TEMPLATE.format(user_query=prompt, context_string=context_string)
                    else:
                        context_string = "\n\n".join(results['documents'][0])
                        final_prompt = PROMPT_TEMPLATE.format(user_query=prompt, context_string=context_string)
                    
                    response_text = llm.generate_content(final_prompt).text
                    
                    if active_agent_name != 'gem5_expert':
                        page_numbers = sorted(list(set([meta.get('page_number') for meta in results['metadatas'][0] if meta.get('page_number')])))
                        if page_numbers: response_text += f"\n\n---\n*Sources: Found on page(s) {', '.join(map(str, page_numbers[:2]))} of the document.*"
                    
                    st.session_state.messages[active_agent_name].append({"role": "assistant", "content": response_text})
                    with st.chat_message("assistant"): st.markdown(response_text)
                except Exception as e: st.error(f"An error occurred while getting the response: {e}")

st.set_page_config(page_title="Multi-Agent Document Platform", layout="centered", page_icon="logo.png")
st.title(" ◆ Multi-Agent Document Platform")

ensure_encoder_exists("gem5_expert")

if "messages" not in st.session_state: st.session_state.messages = {}
if "agent_list" not in st.session_state: 
    st.session_state.agent_list = {
        "gem5_expert": {
            "collection_name": "gem5_documentation_v3", 
            "paths": {
                "classifier": os.path.join(CLASSIFIER_PATH, "gem5_expert_classifier.joblib"),
                "encoder": os.path.join(CLASSIFIER_PATH, "gem5_expert_encoder.joblib")
            },
            "metadata": { 
                "doc_qna_count": "Pre-trained", 
                "casual_qna_count": "Pre-trained",
                "sample_question": "What is the O3 CPU model?"
            }
        }
    } 
if "active_agent" not in st.session_state: st.session_state.active_agent = None
if "page" not in st.session_state: st.session_state.page = "selection"

with st.sidebar:
    if st.session_state.page == "chat":
        if st.button("⬅ Back to Agent Selection", use_container_width=True):
            st.session_state.page = "selection"; st.session_state.active_agent = None; st.rerun()
    st.divider()
    st.subheader("Available Agents")
    for agent_name in list(st.session_state.agent_list.keys()):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            button_type = "primary" if agent_name == st.session_state.active_agent else "secondary"
            if st.button(agent_name, use_container_width=True, key=f"select_{agent_name}", type=button_type):
                st.session_state.active_agent = agent_name; st.session_state.page = "chat"
                st.rerun()
        with col2:
            if agent_name != "gem5_expert" and agent_name !="ML_Book":
                if st.button("🗑️", key=f"delete_{agent_name}", help=f"Delete agent '{agent_name}'"):
                    agent_to_delete = st.session_state.agent_list.pop(agent_name)
                    db_client.delete_collection(name=agent_to_delete["collection_name"])
                    if agent_to_delete.get("paths"):
                        if os.path.exists(agent_to_delete["paths"]["classifier"]):
                            os.remove(agent_to_delete["paths"]["classifier"])
                        if os.path.exists(agent_to_delete["paths"]["encoder"]):
                            os.remove(agent_to_delete["paths"]["encoder"])
                    if agent_name in st.session_state.messages: del st.session_state.messages[agent_name]
                    st.toast(f"Agent '{agent_name}' deleted!")
                    if st.session_state.active_agent == agent_name:
                        st.session_state.page = "selection"; st.session_state.active_agent = None
                    st.rerun()

if st.session_state.page == "selection":
    agent_selection_ui()
else:
    chat_ui()

