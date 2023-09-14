import streamlit as st

from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
import pinecone

from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv  # For loading environment variables from .env file
import os

# Load environment variables from .env file
load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT"))

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat🦜🔗"

@st.cache_resource(ttl="1h")
def configure_retriever():
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


tool = create_retriever_tool(
    configure_retriever(),
    "search_langsmith_docs",
    "Searches and returns documents regarding LangSmith. LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications. You do not know anything about LangSmith, so if you are ever asked about LangSmith you should use this tool.",
)

tools = [tool]

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

# Run the LLM
# print(llm("Tell me a very nice joke"))

# llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")

message = SystemMessage(
    content=(
        "You are a helpful chatbot who is tasked with answering questions about LangSmith. "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about LangSmith. "
        "If there is any ambiguity, you probably assume they are about that."
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)
memory = AgentTokenBufferMemory(llm=llm)
starter_message = "Ask me anything about LangSmith!"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]


def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))