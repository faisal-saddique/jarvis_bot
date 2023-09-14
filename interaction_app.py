import os
from langchain.vectorstores import Pinecone
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv  # For loading environment variables from .env file
import os

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Jarvis Bot", page_icon="🤖")
st.title("Jarvis Bot")

import os
from dotenv import load_dotenv

# Load environment variables from a .env file (contains sensitive information)
load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

@st.cache_resource(ttl="1h")
def configure_retriever():
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    return retriever

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Import Azure OpenAI
from langchain.llms import AzureOpenAI

# Create an instance of Azure OpenAI
# Replace the deployment name with your own
llm = AzureOpenAI(
    deployment_name="Text-Davinci",
    model_name="text-davinci-003",
    temperature=1,
    streaming=True
)

# Setup LLM and QA chain
# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo", temperature=0, streaming=True
# )

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

# if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
#     msgs.clear()
#     msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])