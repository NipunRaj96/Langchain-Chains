from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

prompt=PromptTemplate(
    template="generate 5 interesting fact about {topic}",
    input_variables=['topic']
)

model= ChatHuggingFace(llm=llm)
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic' : 'Delhi'})
print(result)

