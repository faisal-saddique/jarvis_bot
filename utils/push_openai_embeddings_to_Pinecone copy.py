# Import required libraries
from langchain.document_loaders import (
    PyMuPDFLoader,  # For loading PDF files
    DirectoryLoader,  # For loading files from a directory
    TextLoader,  # For loading plain text files
    Docx2txtLoader,  # For loading DOCX files
    UnstructuredPowerPointLoader,  # For loading PPTX files
    UnstructuredExcelLoader  # For loading XLSX files
)

from langchain.document_loaders.csv_loader import CSVLoader  # For loading CSV files
# For splitting text into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# For creating a vector database for similarity search
from langchain.vectorstores import Pinecone
# For generating embeddings with OpenAI's embedding model
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from dotenv import load_dotenv  # For loading environment variables from .env file
import os

# Load environment variables from .env file
load_dotenv()

# Replace with the name of the directory carrying your data
data_directory = "data"

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

# Load your documents from different sources


def get_documents():
    # Create loaders for PDF, text, CSV, DOCX, PPTX, XLSX files in the specified directory
    pdf_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    txt_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.txt", loader_cls=TextLoader)
    csv_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.csv", loader_cls=CSVLoader)
    docx_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.docx", loader_cls=Docx2txtLoader)
    pptx_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader)
    xlsx_loader = DirectoryLoader(
        f"./{data_directory}", glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader)

    # Initialize the 'docs' variable
    docs = None

    # Load files using the respective loaders
    pdf_data = pdf_loader.load()
    txt_data = txt_loader.load()
    csv_data = csv_loader.load()
    docx_data = docx_loader.load()
    pptx_data = pptx_loader.load()
    xlsx_data = xlsx_loader.load()

    # Combine all loaded data into a single list
    docs = pdf_data + txt_data + csv_data + docx_data + pptx_data + xlsx_data

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
embeddings = OpenAIEmbeddings()

docsearch = Pinecone.from_documents(
    docs, embeddings, index_name=os.getenv("PINECONE_INDEX"))

print("Docs pushed to Pinecone.")
