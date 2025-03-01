from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from transformers import AutoTokenizer

db_faiss_path = 'vectorstore/db_faiss'

custom_prompt_template = """Use the given information to answer the user's question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrivel_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bots():
    # Initialize the tokenizer with clean_up_tokenization_spaces
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', clean_up_tokenization_spaces=True)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    
    # Load the FAISS database
    db = FAISS.load_local(db_faiss_path, embeddings, allow_dangerous_deserialization=True)
    
    # Load the language model
    llm = load_llm()
    
    # Set the custom prompt
    qa_prompt = set_custom_prompt()
    
    # Create the QA chain
    qa = retrivel_qa_chain(llm, qa_prompt, db)

    return qa

chain = None

@cl.on_chat_start
async def start():
    global chain  
    chain = qa_bots()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Chatbot, What is your Query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    global chain  

    if chain is None:
        raise ValueError("Chain is not initialized.")
    
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    user_query = message.content 
    res = await chain.ainvoke(user_query, callbacks=[cb])  
    answer = res["result"]

    await cl.Message(content=answer).send()