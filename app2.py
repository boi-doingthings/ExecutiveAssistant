import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import dotenv
dotenv.load_dotenv()

loader = TextLoader("./files/resume.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
persist_dir = "./MyTextEmbedding"

if ~os.path.exists(persist_dir):
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embedding, persist_directory=persist_dir
    )
    vectordb.persist()
else:
    vectordb = None
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)

chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), retriever=vectordb.as_retriever(), chain_type="stuff"
)
query = "Who is Yash Gupta"
chain.run(query)
