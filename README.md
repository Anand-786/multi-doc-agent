# gem5 Code Assistant

An **AI agent** to help anyone in understanding and navigating the **gem5** simulator codebase using natural language.

---

### Motivation

The gem5 simulator is essential for computer architecture research, but its large C++ codebase presents a significant barrier. Finding specific implementations or understanding the flow between different components requires extensive manual effort. This project aims to solve that problem by providing an intelligent agent that can answer questions about the code directly.

### Core Functionality

* **Structural Code Parsing:** Instead of treating code as plain text, this agent uses Abstract Syntax Trees (ASTs) to parse the C++ source. This provides a deep, structural understanding of functions, classes, and their relationships.
* **Intelligent Code Retrieval:** It uses a vector database and semantic search or RAG (Retrieval-Augmented Generation) to find the most relevant code snippets for a user's question.
* **Natural Language Answers:** Leverages a Large Language Model (Google's Gemini) to synthesize the retrieved technical information into a clear, easy-to-understand answer.

### System Architecture

The agent processes a user's question in a few steps.

1.  First, the user's question is converted into a numerical vector representation.
2.  This vector is used to search a pre-processed database of the gem5 codebase to find the most relevant code snippets. This database is built beforehand by parsing the entire codebase using ASTs and storing each function/class as a vector.
3.  Finally, these relevant code snippets and the original question are sent to the Gemini LLM, which generates a comprehensive, human-readable answer.

### Technology Stack

* **Backend:** Python
* **AI/ML:** Google Generative AI (Gemini), SentenceTransformers, ChromaDB
* **Code Parsing:** Clang (libclang)

### Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPONAME.git](https://github.com/YOUR_USERNAME/YOUR_REPONAME.git)
    cd YOUR_REPONAME
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Key:**
    * Create a `.env` file in the project root.
    * Add your API key to the file: `GOOGLE_API_KEY="YOUR_API_KEY_HERE"`

5.  **Build the Database:**
    * Place the gem5 codebase into the `data/` directory.
    * Run the ingestion script:
    ```bash
    python src/ingest.py
    ```

6.  **Run the Agent:**
    ```bash
    python src/agent.py
    ```