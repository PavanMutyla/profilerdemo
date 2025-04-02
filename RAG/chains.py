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
    ("human", "Retrieved document: \n\n {documents} \n\n User question: {question}")
])

retrieval_grader = grade_prompt | structured_llm_grader


prompt = PromptTemplate(
    template='''
    You are a Registered Investment Advisor with expertise in Indian financial markets and client relations.
    You must understand what the user is asking about their financial investments and respond to their queries based on the information in the documents only.
    Use the following documents to answer the question. If you do not know the answer, say you don't know.
    Query: {question}
    Documents: {context}
    ''',
    input_variables=['question', 'context']
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
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


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
            "{categories}\n\nReturn a JSON object like:\n"
            '{"Product Categories": 0.3, "Investment Regulations": 0.4, "Taxation Data": 0.2, "Market Segments": 0.1, "Cultural Aspects": 0.0}',
        ),
    ]
)

query_classifier = classification_prompt | llm | StrOutputParser()

