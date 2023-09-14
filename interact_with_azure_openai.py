from dotenv import load_dotenv

load_dotenv()

# Import Azure OpenAI
from langchain.llms import AzureOpenAI

# Create an instance of Azure OpenAI
# Replace the deployment name with your own
llm = AzureOpenAI(
    deployment_name="Text-Davinci",
    model_name="text-davinci-003",
    temperature=1
)

# Run the LLM
print(llm("Tell me a very nice joke"))

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(deployment="Text-Davinci")

print(embeddings.embed_query("hello world!"))