from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
import os


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    general_kb = PineconeVectorStore(
         pinecone_api_key=os.environ.get('PINCEONE_API_KEY'),
        embedding=embedding_model,
        index_name='rag-rubic',
        namespace='vectors_lightmodel'

    )
    general_retriever = general_kb.as_retriever(search_kwargs={"k": 10})
except Exception as e:
    print(f"Pinecone connection error: {e}")
    # Fallback to SKLearn vector store if Pinecone fails
    general_retriever = None

try:
    regulations_kb = PineconeVectorStore(
         pinecone_api_key=os.environ.get('PINCEONE_API_KEY'),
        embedding=embedding_model,
        index_name='rag-rubic',
        namespace='regulations')
    regulations_retriever = regulations_kb.as_retriever(search_kwargs= {"k":10})
except Exception as e:
    print(f"Pinecone connection error: {e}")
    # Fallback to SKLearn vector store if Pinecone fails
    regulations_retriever = None

try:
    fin_products_kb = PineconeVectorStore(
         pinecone_api_key=os.environ.get('PINCEONE_API_KEY'),
        embedding=embedding_model,
        index_name='rag-rubic',
        namespace='financial_products')
    fin_prods_retriever = fin_products_kb.as_retriever(search_kwargs= {"k":10})
except Exception as e:
    print(f"Pinecone connection error: {e}")
    # Fallback to SKLearn vector store if Pinecone fails
    fin_products_retriever = None

try:
    tax_kb = PineconeVectorStore(
         pinecone_api_key=os.environ.get('PINCEONE_API_KEY'),
        embedding=embedding_model,
        index_name='rag-rubic',
        namespace='tax_data')
    tax_retriever = tax_kb.as_retriever(search_kwargs= {"k":10})
except Exception as e:
    print(f"Pinecone connection error: {e}")
    # Fallback to SKLearn vector store if Pinecone fails
    tax_retriever = None
