import os
import dotenv

dotenv.load_dotenv()

# from langchain.document_loaders import OnlinePDFLoader
# loader = OnlinePDFLoader("https://drive.google.com/file/d/1h0VFtINFoqiNf1DJ8rpCWTyZB7l01peB/view")

import requests
from langchain.document_loaders import TextLoader

loader = TextLoader("./files/resume.txt")
documents = loader.load()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

auth_config = weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"])

client = weaviate.Client(
    url="https://my-personal-bot-bxmogd3i.weaviate.network",
    auth_client_secret=auth_config,
    #   embedded_options = EmbeddedOptions()
)
vectorstore = Weaviate.from_documents(
    client=client, documents=chunks, embedding=OpenAIEmbeddings(), by_text=False
)

retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)

from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "Who is Yash?"
rag_chain.invoke(query)
