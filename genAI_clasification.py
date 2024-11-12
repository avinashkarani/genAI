#%%
# Import relevant functionality
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
import getpass
import os
import configparser
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.exceptions import LangChainError
#from langchain_chroma import Chroma
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# Read API keys from config file
config = configparser.ConfigParser()
config.read('config.ini')

os.environ["TAVILY_API_KEY"] = config['API_KEYS']['TAVILY_API_KEY']
os.environ["OPENAI_API_KEY"] = config['API_KEYS']['OPENAI_API_KEY']

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# loader = CSVLoader(file_path="./taxanomy.csv")
# data = loader.load()
try:
    tax = pd.read_csv('./taxanomy.csv')
except FileNotFoundError:
    raise FileNotFoundError("The file 'taxanomy.csv' was not found. Please check the file path.")

#%%
# db = Chroma.from_documents(data, embedding_function=embeddings)
# vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings,
#     persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
# )

#%%
# Create the agent
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    raise FileNotFoundError("The file 'data.csv' was not found. Please check the file path.")

df.dropna(how='all', inplace=True)
df.drop_duplicates(inplace=True)

# Ensure required columns are present
required_columns = ['Vendor Name', 'Country', 'City', 'Address']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df[required_columns]

#%%
prompt = (
    "You are a helpful assistant. "
    "You may not need to use tools for every query - the user may just want to chat!"
)
memory = MemorySaver()
model = ChatOpenAI(model_name="gpt-4o")
search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools,state_modifier=prompt,debug=False)
# Convert each row to a string
row_strings = df.apply(lambda row: row.to_string(), axis=1)
# prompt=
for row_str in row_strings:
    try:
        result=agent_executor.invoke({"messages": [HumanMessage(content= f"Retrieve the parent company or related entities for {row_str}. Additionally, identify the primary business Top Category Name,Category Code,Category Name of the company only from the below table mapping {tax.to_string()},if exact match is not availabe please provide close one. Provide the output in the following simplified format: Company Name: [Original Company Name] Parent Company: [Parent Company Name or 'None' if independent]  Category: [Top  Category] Category code: [Category code]")]})
        result["messages"][-1].pretty_print()
    except LangChainError as e:
        print(f"An error occurred while processing the row: {row_str}")
        print(f"Error: {e}")