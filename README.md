# Machine Learning ChatBot

A document-based RAG chatbot built with Python, Streamlit, LangChain, ChromaDB, and Mistral AI. It loads your local documents into a vector store and answers questions using only the retrieved context from those files.

## Features

- Streamlit web interface for chatting with your documents
- Retrieval-Augmented Generation (RAG) pipeline
- Vector database using ChromaDB
- Embeddings with Hugging Face models
- Mistral-powered answer generation
- Persistent local vector store in the project folder

## Project Structure

```text
Machine Learning ChatBot/
├── app.py                  # Streamlit UI version
├── main.py                 # CLI-based RAG script
├── create_database.py      # Database creation helper
├── requirements.txt        # Python dependencies
├── .env                    # Local environment variables (not included by default)
├── chroma_db/              # Persisted vector database
├── document loader/        # Document ingestion utilities
├── vector stores/          # Vector store logic
└── README.md               # Project documentation
```

## Tech Stack

- Python 3.10+
- Streamlit
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Mistral AI
- PDF / DOCX document parsing support

## Setup

1. Open the project folder in your terminal.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Mistral API key:

```env
MISTRAL_API_KEY=your_api_key_here
```

## Run the App

### Streamlit UI

```bash
streamlit run app.py
```

### CLI Version

```bash
python main.py
```

## How It Works

1. Documents are loaded from the project folder or relevant file sources.
2. Text is chunked and embedded using Hugging Face embeddings.
3. ChromaDB stores the embeddings for semantic retrieval.
4. When you ask a question, the retriever fetches relevant document chunks.
5. The Mistral model answers using only the retrieved context.

## Notes

- The vector database is persisted under `chroma_db`, so you can reuse previously indexed content.
- If you add or update documents, regenerate the vector database if necessary.
- The app is designed for local document Q&A and works best with structured, searchable content.

## License

This project is provided as-is for learning and local development purposes.
