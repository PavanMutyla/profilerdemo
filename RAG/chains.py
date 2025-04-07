"""All prompts utilized by the RAG pipeline"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os

llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=os.environ.get('OPEN_AI_KEY'),
    temperature=0.2
)

# Schema for grading documents
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)
system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document: \n\n {data} \n\n User question: {query}")
])

retrieval_grader = grade_prompt | structured_llm_grader


prompt = PromptTemplate(
    template='''
    You are a Registered Investment Advisor with expertise in Indian financial markets and client relations.
    You must understand what the user is asking about their financial investments and respond to their queries based on the information in the documents and user's financial and demographic profile only.
    Prioritize answer based on the user's profile and if there is not much profile data, prioritize user's query and answer.
    Use the following documents to answer the question. If you do not know the answer, say you don't know.
    Query: {query}
    Documents: {data}
    User's Data : {user_data}
    ''',
    input_variables=['query', 'data', 'user_data']
)

rag_chain = prompt | llm | StrOutputParser()


# Prompt
system_rewrite = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        (
            "human",
            "Here is the initial question: \n\n {query} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


from pydantic import BaseModel, Field, RootModel
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser

# Define the Pydantic model using RootModel
class CategoryProbabilities(RootModel):
    """Probabilities for different knowledge base categories."""
    root: Dict[str, float] = Field(description="Dictionary mapping category names to probability scores")

system_classifier = """You are a query classifier that determines the most relevant knowledge bases (KBs) for a given user query. 
Analyze the semantic meaning and intent of the query and assign probability scores (between 0 and 1) to each KB.

Ensure the probabilities sum to 1 and output a JSON dictionary with category names as keys and probabilities as values.
"""

classification_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_classifier),
        (
            "human",
            "Here is the user query: \n\n {query} \n\n Assign probability scores to each of the following KBs:\n"
            "{categories}\n\nReturn a JSON object with category names as keys and probability scores as values."
        ),
    ]
)

# Create a JSON output parser
json_parser = JsonOutputParser(pydantic_object=CategoryProbabilities)

# Create the chain with the structured output parser
query_classifier = classification_prompt | llm | json_parser


#query_classifier = classification_prompt | llm | StrOutputParser()

