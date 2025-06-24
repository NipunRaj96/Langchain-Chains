from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)
str_parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='classify the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='classify the sentiment of the following text into positive or negative: \n{text} \n{format_instructions}',
    input_variables=['text'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='write an appropriate reply for this positive feedback \n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='write an appropriate reply for this negative feedback \n{text}',
    input_variables=['text']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | str_parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | str_parser),
    RunnableLambda(lambda x: "could not classify sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({
    'text': 'While the product quality was decent and it arrived on time, the overall experience was frustrating due to poor customer support and misleading return policies.'
})

print(result)

chain.get_graph().print_ascii()