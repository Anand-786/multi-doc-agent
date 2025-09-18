# Gem5 Documentation Agent

An agent that uses Retrieval-Augmented Generation (RAG) to answer questions about the gem5 computer architecture simulator. It uses a vector database to find relevant information from the official gem5 documentation before generating an answer.

**Deployment** : [gem5-doc-agent](https://gem5-doc-agent.streamlit.app/)

---

## Features

- **Intent Classification:** A simple classifier determines if a question is about gem5 or a general topic. This acts as the agent's primary decision-making step.
- **RAG for Technical Questions:** For gem5-related queries, the agent uses its RAG tool, retrieving relevant text from a ChromaDB vector database to provide context to the language model.
- **General Q&A:** For casual or off-topic questions, the agent queries the language model directly, using its general knowledge.
- **Simple Chat Interface:** A basic user interface built with Streamlit.

---
## Application Workflow

When a user submits a query, the agent follows a specific sequence of steps:

1.  **Intent Classification:** The raw query is first passed to a trained Logistic Regression model. This classifier determines if the query is `doc_qna` (related to gem5) or `casual` (a general question).
2.  **Conditional Routing:** The application logic then branches based on the classified intent.
    -   **If the intent is `casual`**, the query is sent directly to the Gemini LLM. The model answers using its general knowledge, and the process ends here.
    -   **If the intent is `doc_qna`**, the RAG pipeline is triggered.
3.  **Retrieval (RAG):** The user's query is converted into a vector embedding and used to search the ChromaDB database. The top 5 most similar text chunks from the gem5 documentation are retrieved.
4.  **Generation (RAG):** The retrieved text chunks are combined with the original query into a detailed prompt. This is sent to the Gemini LLM, which is instructed to formulate an answer based only on the provided context.
5.  **Display:** The final response from either path is displayed in the chat interface.

---

## Tech Stack

- **Intent Classifier:** Scikit-learn (Logistic Regression)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** Google Gemini 2.5 Flash
- **Vector Database:** ChromaDB
- **Frontend:** Streamlit

---

## Setup and Local Installation

To run this application on your local machine, follow these steps.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Anand-786/gem5-doc-agent.git](https://github.com/Anand-786/gem5-doc-agent.git)
    cd gem5-doc-agent
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a file named `.env` in the project's root directory and add your Google API key:
    ```
    GOOGLE_API_KEY="your_api_key"
    ```

5.  **Run the Application**
    ```bash
    streamlit run src/app.py
    ```

---

## Project Components

- `app.py`: The main Streamlit application.
- `train_router.py`: Script to train the intent classification model.
- `intent_dataset.csv`: The data used to train the intent classifier.
- `gem5_chroma_db_v3/`: The ChromaDB vector store containing the documentation embeddings.
- `requirements.txt`: Python package dependencies.

---

## Screenshot

![Gem5 Agent Screenshot](assets/screenshot.png)