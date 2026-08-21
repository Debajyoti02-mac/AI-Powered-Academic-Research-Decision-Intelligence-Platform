from mcp.server.fastmcp import FastMCP
mcp = FastMCP(name="edumind")

import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf = PyPDFLoader("Why_Language_Models_Hallucinate_Explainer.pdf")
document = pdf.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=120, chunk_size=1200)
pages = text_splitter.split_documents(documents=document)
chunks = [i.page_content for i in pages]

print(f"Total chunks: {len(chunks)}")


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./Local_data")

collection = client.get_or_create_collection(
    name="edumind",
    embedding_function=embedding_function
)

if collection.count() == 0:
    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))],
        metadatas=[i.metadata for i in pages]
    )

print(f"Collection size: {collection.count()}") 

from rank_bm25 import BM25Okapi

def tokenize(text: str):
    return text.lower().split()

tokenized_chunks = [tokenize(c) for c in chunks]
corpus_build = BM25Okapi(tokenized_chunks)


from langchain_groq import ChatGroq
import os 
from dotenv import  load_dotenv 
load_dotenv()
key = os.getenv("GROQ_API_KEY")
chat_model = ChatGroq(model="openai/gpt-oss-120b")

# rewritten_query
@mcp.tool()
def Retrival(query:str):
    prompt = f""" make the prompt contextual : {query} """
    rewritten_query = chat_model.invoke(prompt).content

    response = collection.query(query_texts=[rewritten_query], n_results=5)
    distances = response['distances'][0]
    documents = response['documents'][0]

    threshold = 0.8
    vector_chunks = [doc for dist, doc in zip(distances, documents) if dist < threshold]

    scores = corpus_build.get_scores(tokenize(rewritten_query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    bm25_chunks = [chunks[i] for i, _ in ranked[:10]]   # bug fix: index -> actual text

    rrf_scores = {}
    for rank, doc in enumerate(vector_chunks):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (rank + 60)
    for rank, doc in enumerate(bm25_chunks):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (rank + 60)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in merged[:5]]

    context_text = "\n\n".join(top_docs) if top_docs else ""
    return {"context": context_text}

import numexpr
@mcp.tool()
def calculator(expression:str):
    """ anytypes of aithmetic oparations is done by this tool """ 
    try : 
        return str(numexpr.evaluate(expression).item())
    except Exception as e :
        return str(e)

from tavily import TavilyClient

@mcp.tool()
def web_search(query:str)->str:
    """ user qustions fallback to this tool """ 
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query=query)
    return str(response)


import os 
@mcp.tool()
def file_read(filename:str): 
    """ user given filename content read by this tool """
    try:
        with open (file=filename) as f :
            return f"content is : {f.read()}"
    except Exception as e:
        return str(e)
@mcp.tool()
def file_write(filename:str ,content:str):
    """ user given file path read the content if file doesnt exist then create and write the content """
    os.makedirs(os.path.dirname(os.path.abspath(filename)),exist_ok=True) 
    try: 
        with open (file=filename , mode="w") as file :
            content_write= file.write(content)
            return f"sucessfull {content_write}"
    except Exception as e :
        return str(e)

if __name__ == "__main__":
    mcp.run()