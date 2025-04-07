import os
import re
import markdown
import numpy as np
from IPython.display import Image
from typing import Dict
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
from typing import List, TypedDict, Optional, Annotated, Sequence
from langchain.schema import Document
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
from RAG.entities import KBCategory, THRESHOLD
from RAG.mappings import CATEGORY_DB_MAPPING
from RAG.chains import retrieval_grader, query_classifier, question_rewriter, rag_chain
from RAG.kb import tax_retriever, fin_prods_retriever, general_retriever, regulations_retriever
from langchain_openai import OpenAIEmbeddings
import json

load_dotenv()


vector_dbs = {
    "tax_retriever": tax_retriever,
    "fin_prods_retriever": fin_prods_retriever,
    "general_retriever": general_retriever,
    "regulations_retriever": regulations_retriever
}
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=os.environ.get('OPEN_AI_KEY'),
    temperature=0.2
)

llm_google =  ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

web_search_tool = TavilySearchResults(api_key=os.environ.get('TAVILY_API_KEY'), k=10)




class GraphState(TypedDict):
    user_data: Optional[Dict]
    query: str
    generation: Optional[str]
    web_search: Optional[str]  
    data: List
    relevant_categories : List
    rewritten : bool
    action : Optional[str]

def classify_query(state, chain=query_classifier):
    query = state['query']
    user_data = state.get('user_data', {})
    kb_categories = {cat.value: cat for cat in KBCategory}  # Map values to KBCategory members
    
    try:
        # Get parsed result directly from the chain - this will be a dictionary
        parsed_result = chain.invoke({"query": query, "categories": list(kb_categories.keys())})
        
        threshold_value = THRESHOLD.threshold.value
        
        # Select only categories where probability >= threshold and return the corresponding KBCategory
        relevant_categories = [kb_categories[category] for category, prob in parsed_result.items() 
                              if prob >= threshold_value]
        
        # If no categories meet the threshold, default to using the general category
        if not relevant_categories and "general" in kb_categories:
            relevant_categories = [kb_categories["general"]]
            
    except Exception as e:
        print(f"Error classifying query: {e}")
        # Default to general category in case of any error
        relevant_categories = [kb_categories["general"]] if "general" in kb_categories else []
    
    return {'data': [], 'query': query, 'relevant_categories': relevant_categories, 'web_search': 'yes', 'user_data': user_data}





def fetch_data(state, vector_dbs = vector_dbs, tailvy_api_client=web_search_tool) -> dict:
    """
    Fetch relevant data based on category classification.
    If a category maps to multiple retrievers, fetch data from all.
    """
    query = state['query']
    user_data = state.get('user_data', {})
    relevant_categories = state['relevant_categories']  # Get categories above the threshold
    results = {}

    for category in relevant_categories:
        db_keys = CATEGORY_DB_MAPPING.get(KBCategory(category), [])  # Get list of retriever names
       

        category_results = []  
        for db_key in db_keys:
            if db_key == "tailvy_api":
                category_results.append(tailvy_api_client.invoke({"query": query}))  # Fetch from web search
            elif db_key in vector_dbs:
                category_results.append(vector_dbs[db_key].invoke(query))  # Fetch from the retriever
            else:
                category_results.append("No data source available")

        results[category] = category_results  # Store all results for the category

    return {'data': results, 'query':query,'relevant_categories':relevant_categories, 'web_search':'yes', 'user_data': user_data}

def rewrite_query(state, rewrite_chain=question_rewriter):
    query = state['query']
    user_data = state.get('user_data', {})
    # Get the current rewrite count or initialize to 0
    rewrite_count = state.get('rewrite_count', 0)
    
    # Increment the rewrite count
    rewrite_count += 1
    
    rewritten_query = rewrite_chain.invoke({'query': query})

    return {
        'data': [],
        'query': rewritten_query,
        'relevant_categories': state['relevant_categories'],
        'web_search': 'no',
        'rewritten': True,
        'rewrite_count': rewrite_count,  # Store the updated count
        'user_data': user_data
    }


def grade_data(state):
    """Grades the docs generated by the retriever_db
    If 1, returns the docs if 0 proceeds for web search"""
    question = state['query']
    user_data = state.get('user_data', {})
    docs = state['data']
    filterd_data = []
    web = "no"
    for data in docs:
        score = retrieval_grader.invoke({'query':question, 'data':docs})
        
        grade = score.binary_score
       
        if grade == 'yes':
            filterd_data.append(data)
        else:
            #print("----------Failed, proceeding with WebSearch------------------")
            web = 'yes'
    return {"documents": filterd_data, "question": question, "web_search": web, 'user_data': user_data}



def decide(state):
    """Decide if the generation should be based on DB, web search, or if query should be rewritten"""
    web = state.get('web_search', 'no')
    user_data = state.get('user_data', {})
    # Get rewrite count or default to 0
    rewrite_count = state.get('rewrite_count', 0)
    # Set maximum number of rewrites
    MAX_REWRITES = 1
    
    # If data is insufficient, no web search is needed, and we haven't hit the rewrite limit
    if web == 'no' and (not state.get('data') or len(state.get('data', [])) == 0) and rewrite_count < MAX_REWRITES:
        return 'rewrite_query'
    # If we've hit the rewrite limit or web search is needed, go to web search
    elif web == 'yes' or rewrite_count >= MAX_REWRITES:
        return 'perform_web_search'
    else:
        return 'generate'







    
def web_search(state,tailvy_api_client=web_search_tool ):
    query = state['query']
    user_data = state.get('user_data', {})
    data = state['data']
    result = tailvy_api_client.invoke({"query": query})
    docs = []
    for res in result:
        content = res["content"]  # Extract answer content
        source = res["url"]       # Extract source URL
        
        # Create Document with metadata
        doc = Document(page_content=content, metadata={"source": source})
        docs.append(doc)

    if not result:
        #print("No results from web search. Returning default response.")
        return {"documents": [], "question": query}

    if isinstance(data, dict):
        data = []  # Initialize data as an empty list if it's a dictionary

    data.extend(docs)
    return {'data':data, 'query':query,  'relevant_categories':state['relevant_categories'], 'user_data': user_data}



def rerank(state):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get('OPEN_AI_KEY'))
    query = state['query']
    user_data = state.get('user_data', {})
    data = state['data']
    
    # If there's no data to rerank, just return the original state
    if not data:
        return state
    
    # Create a list to store reranked documents
    reranked_docs = []
    
    # Check the structure of data to determine how to process it
    if isinstance(data, list):
        # If data is already a list of documents
        documents = data
    elif isinstance(data, dict):
        # If data is a dictionary (possibly with categories as keys)
        # Flatten all documents from all categories into a single list
        documents = []
        for category, docs_list in data.items():
            if isinstance(docs_list, list):
                for doc_group in docs_list:
                    if isinstance(doc_group, list):
                        documents.extend(doc_group)
                    else:
                        documents.append(doc_group)
    
    # Only proceed if we have documents to rank
    if documents:
        # Get query embedding
        query_embedding = np.array(embeddings.embed_query(query))
        
        # Calculate similarity scores for each document
        doc_scores = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
               
                # If it's a Document object with page_content
                content = doc.page_content
            elif isinstance(doc, dict) and 'content' in doc:
                # If it's a dictionary with a 'content' key
                content = doc['content']
            elif isinstance(doc, dict) and 'page_content' in doc:
                # If it's a dictionary with a 'page_content' key
                content = doc['page_content']
            elif isinstance(doc, str):
                # If it's a string
                content = doc
            else:
                # Skip if we can't extract content
                continue
                
            # Get document embedding and calculate similarity
            doc_embedding = np.array(embeddings.embed_query(content))
            similarity = np.dot(query_embedding, doc_embedding)
            doc_scores.append((doc, similarity))
        
        # Sort documents by similarity score in descending order
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in doc_scores]
    
    # Return updated state with reranked documents
    return {
        'data': reranked_docs,
        'query': query,
        'relevant_categories': state.get('relevant_categories', []),
        'user_data': user_data
    }





def generate(state, rag_chain=rag_chain):
    query = state['query']
    data = state['data']
    user_data = state.get('user_data', {})

    response = rag_chain.invoke({
        'data': data,
        'query': query,
        'user_data': user_data
    })

    # Remove repeated intro if present
    response = re.sub(
        r'^Here are some low-risk investment areas you might consider:.*?\d+\.\s',
        '',
        response,
        flags=re.DOTALL
    )

    investment_options = re.split(r'\n?\d+\.\s+', response.strip())
    investment_options = [opt for opt in investment_options if opt.strip()]

    markdown_response = "**Here are some low-risk investment areas you might consider:**\n\n"

    for i, option in enumerate(investment_options, start=1):
        match = re.match(r'\*\*(.*?)\*\*[:-]?\s*(.*)', option, re.DOTALL)
        if match:
            title, desc = match.groups()
            desc = desc.strip().replace('\n', ' ')
            markdown_response += f"**{i}. {title}**\n- {desc}\n\n"
        else:
            markdown_response += f"**{i}.** {option.strip()}\n\n"

    # Return the markdown directly instead of converting to HTML
    return {
        'question': query,
        'generation': markdown_response  # Return markdown formatted text directly
    }






# Update the workflow with the new node and connections
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("classify_query", classify_query)
workflow.add_node("fetch_data", fetch_data)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("grade_data", grade_data)
workflow.add_node("decide", decide)
workflow.add_node("perform_web_search", web_search)
workflow.add_node("rerank", rerank)
workflow.add_node("generate", generate)

# Define edges
workflow.add_edge(START, "classify_query")
workflow.add_edge("classify_query", "fetch_data")
workflow.add_edge("fetch_data", "grade_data")

workflow.add_conditional_edges("grade_data", 
                             decide,{
                                 "perform_web_search": "perform_web_search",
                                 "generate": "rerank",  # Change destination to rerank instead of generate
                                 "rewrite_query": "rewrite_query"
                             })

# Add a connection from rewrite_query back to fetch_data
workflow.add_edge("rewrite_query", "fetch_data")
workflow.add_edge("perform_web_search", "rerank")  # Web search also goes to rerank first
workflow.add_edge("rerank", "generate")  # Rerank then goes to generate
workflow.add_edge("generate", END)

# Compile the graph
graph = workflow.compile()


'''

png_graph = graph.get_graph().draw_mermaid_png()
with open("my_graph.png", "wb") as f:
    f.write(png_graph)

print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")



# Assuming you have a query and have called classify_query
query = "What are the financial products available in India?"
inputs = {"query": query}
# Fetch the relevant database retriever
result =  graph.invoke(inputs)
print(result['generation'])
'''