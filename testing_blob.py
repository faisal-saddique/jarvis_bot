from langchain.document_loaders import AzureBlobStorageContainerLoader
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

OpenAI()
load_dotenv()

loader = AzureBlobStorageContainerLoader(conn_str=os.getenv("AZURE_BLOB_CONN_STR"), container=os.getenv("AZURE_BLOB_CONTAINER"))

docs = loader.load()

print(docs)