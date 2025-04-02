import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, TypedDict, Optional
from langchain.schema import Document
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
from RAG.entities import KBCategory
from RAG.chains import retrieval_grader, query_classifier, question_rewriter, rag_chain
from RAG.kb import tax_retriever, fin_prods_retriever, general_retriever, regulations_retriever
load_dotenv()


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=os.environ.get('OPEN_AI_KEY'),
    temperature=0.2
)

llm_google =  ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')


def classify_quer(query, chain = query_classifier):
    kb_categories = [cat.value for cat in KBCategory]
    result = chain.invoke({"query": query, "categories": kb_categories})
    return result

