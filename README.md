# Company Information RAG System

## Project Description

This project is a Retrieval-Augmented Generation (RAG) application designed to answer questions about company information.

It retrieves relevant information from company documents using semantic similarity search and uses Google's Gemini AI model to generate natural-language answers based on the retrieved context.

## Technologies Used

- **Python** – Main programming language.
- **LangChain** – Connects the different components of the RAG pipeline.
- **HuggingFace** – Generates text embeddings using `all-MiniLM-L6-v2`.
- **ChromaDB** – Stores embeddings and performs similarity searches.
- **Google Gemini** – Generates answers using retrieved information.
- **Sentence Transformers** – Supports the embedding model.

## Project Structure

```text
rag-project/
│
├── documents/          # Original company documents
├── chroma_db/          # Persistent vector database
├── venv/               # Python virtual environment
├── .env                # API keys and secrets
├── .gitignore          # Files excluded from Git
├── requirements.txt    # Python dependencies
├── rag_pipeline.py     # Main RAG application
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate into the project:

```bash
cd rag-project
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_GENAI_API_VERSION=v1beta
```

Never upload your `.env` file to GitHub.

## Running the Application

Run the main pipeline:

```bash
python rag_pipeline.py
```

The application will load the company documents, generate embeddings, store them in ChromaDB, and allow users to ask questions.

## How It Works

1. Company documents are loaded.
2. Documents are divided into smaller chunks.
3. HuggingFace converts chunks into numerical embeddings.
4. ChromaDB stores the embeddings and associated text.
5. A user's question is converted into an embedding.
6. ChromaDB retrieves the most relevant chunks.
7. Gemini generates an answer using the retrieved context.

## Author

Personal AI Engineering Project.