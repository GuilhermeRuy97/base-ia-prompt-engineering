from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

load_dotenv()
 
langfuse_handler = CallbackHandler()
 
llm = ChatOpenAI(model_name="gpt-4o")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm
 
response = chain.invoke(
    {"topic": "cats"}, 
    config={"callbacks": [langfuse_handler]})