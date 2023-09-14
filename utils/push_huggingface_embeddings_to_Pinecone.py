# Import required libraries
from langchain.document_loaders import (
 AzureBlobStorageContainerLoader
)

# For splitting text into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# For creating a vector database for similarity search
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from dotenv import load_dotenv  # For loading environment variables from .env file
import os

# Load environment variables from .env file
load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

def get_documents():
    loader = AzureBlobStorageContainerLoader(conn_str=os.getenv("AZURE_BLOB_CONN_STR"), container=os.getenv("AZURE_BLOB_CONTAINER"))
    docs = loader.load()
    # Return all loaded data
    return docs


# Get the raw documents from different sources
raw_docs = get_documents()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=5)

docs = text_splitter.split_documents(raw_docs)

print(f"Total docs: {len(docs)}")

# Create OpenAIEmbeddings object using the provided API key
embeddings = HuggingFaceEmbeddings()

docsearch = Pinecone.from_documents(
    docs, embeddings, index_name=os.getenv("PINECONE_INDEX"))

print("Docs pushed to Pinecone.")