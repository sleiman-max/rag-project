
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Find the documents folder
DOCUMENTS_PATH = Path("documents")

# Load all text files from the folder
loader = DirectoryLoader(
    str(DOCUMENTS_PATH),
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()

print(f"Loaded {len(documents)} document(s).")

for document in documents:
    print(document.page_content[:300])

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
print("CHUNKING CODE IS RUNNING!")
print(f"\nCreated {len(chunks)} chunks.")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)

from langchain_huggingface import HuggingFaceEmbeddings

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert the first chunk into a numerical vector
sample_vector = embeddings.embed_query(chunks[0].page_content)

print("\nEmbedding created successfully!")
print(f"Vector dimensions: {len(sample_vector)}")
print(f"First 10 numbers: {sample_vector[:10]}")
from langchain_chroma import Chroma

# Create a persistent Chroma vector database
vector_store = Chroma(
    collection_name="company_information",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Delete old collection to prevent duplicate chunks
vector_store.delete_collection()

# Create a fresh collection
vector_store = Chroma(
    collection_name="company_information",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Store document chunks
vector_store.add_documents(chunks)

print("\nDocuments successfully stored in ChromaDB!")
print(f"Number of stored chunks: {vector_store._collection.count()}")

print("\nDocuments successfully stored in ChromaDB!")
print(f"Number of stored chunks: {vector_store._collection.count()}")
# Test similarity search
query = input("\nAsk a question about the company: ")

results = vector_store.similarity_search(query, k=2)

print("\nSEARCH RESULTS:")

for result in results:
    print("\n", result.page_content)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0
)

# Combine retrieved information
context = "\n\n".join(
    document.page_content for document in results
)

# Create the RAG prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful company assistant.

Answer the question using ONLY the information provided below.

If the information is not available, say:
"I don't have that information in my knowledge base."

Context:
{context}

Question:
{question}
""")

# Generate the answer
chain = prompt | llm

response = chain.invoke({
    "context": context,
    "question": query
})

print("\nFINAL RAG ANSWER:")
print(response.text)