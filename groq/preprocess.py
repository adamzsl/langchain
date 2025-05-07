import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS # Facebook AI Similarity Search - efficient similarity search and clustering of vectors
from pathlib import Path


load_dotenv()

# Load & split
loader = WebBaseLoader("https://docs.smith.langchain.com/prompt_engineering/quickstarts/quickstart_sdk") # URL of the documentation site
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Split the documents into smaller chunks
chunks = splitter.split_documents(docs[:50]) # Limit to 50 documents for testing

# Embed & index
embeddings = OllamaEmbeddings(model="llama3.2") # Embedding model - for text representation
vectorstore = FAISS.from_documents(chunks, embeddings) # Create a FAISS index from the documents and embeddings

# Save the FAISS index to a local directory
out_dir = Path(__file__).parent / "faiss_store"
vectorstore.save_local(str(out_dir))
print(f"Saved FAISS index to {out_dir.resolve()}")