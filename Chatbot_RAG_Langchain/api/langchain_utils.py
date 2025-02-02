from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from chroma_utils import vectorstore
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 6})

output_parser = StrOutputParser()

# Set up prompts and chains
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])



qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """Follow the instructions: 1. You are a professional consultant working in a HR consulting and training company called Precena Strategic Partners.
     2. Only if the human greets you, respond with 'Welcome to Precena Strategic Partners. How can I help you today?' Otherwise, ignore the greeting.
     3. If a name is provided, use it in your response. If no name is provided, ignore it.
     4. Refer only to the context provided to answer the question.
     5. Use UK English, and a professional and helpful tone while responding.
     6. Use the most relevant framework provided in the context to explain your point.
     7. Also provide examples after you explain your point. Examples should preferably be from the context provided. If no examples are available, you can use examples from other training data, but avoid using sensitive topics like gender, ethinicity and politics.
     8. Do not respond in a rude or unprofessional tone ever, no matter what the human says. Also never provide personal opinions or experiences. Only provide professional advice and information.
     9. Do not use the word 'context' in your response. Instead, frame it as 'information from Precena'.
     10. Do not include information that is not in the context provided.
     11. End with 'is there anything else you would like to know?'"""),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])



def get_rag_chain(model="gpt-4o-mini"):
    llm = ChatOpenAI(model=model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain
