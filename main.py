import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from prompt_config import system_prompt

current_script_path = Path(__file__).resolve().parent
current_script_path = os.path.dirname(current_script_path)
current_script_path = os.path.join(current_script_path, "VLM")
faiss_index_path = os.path.join(current_script_path, "faiss_index")
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

def embedding_storage(embeddings, description, username = "index"):
    # Initialize Vectorstore
    faiss_index_path_file = os.path.join(faiss_index_path, f"{username}.faiss")

    if not os.path.exists(faiss_index_path_file):
        texts = ['ignore this text']
        db = FAISS.from_texts(texts, embeddings)
        db.save_local(folder_path=str(faiss_index_path))

    vectorstore = FAISS.load_local(str(faiss_index_path), embeddings, allow_dangerous_deserialization=True, index_name = username) # Default: index.faiss.you can change the index name by passing the variable: FAISS.load_local(str(faiss_index_path), embeddings, index = Username) 
    retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=0.01, search_kwargs=dict(k=4))
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    # history = memory.load_memory_variables({"prompt": question})["history"])
    memory.save_context({"input": "describe the image"},{"output": description})
    vectorstore.save_local(folder_path=str(faiss_index_path), index_name = username)

    # return role_response

def RAG_pipeline(llm, embeddings, question, username = "index"):
    # Initialize Vectorstore
    faiss_index_path_file = os.path.join(faiss_index_path, f"{username}.faiss")

    if not os.path.exists(faiss_index_path_file):
        texts = ['ignore this text']
        db = FAISS.from_texts(texts, embeddings)
        db.save_local(folder_path=str(faiss_index_path))

    vectorstore = FAISS.load_local(str(faiss_index_path), embeddings, allow_dangerous_deserialization=True, index_name = username) # Default: index.faiss.you can change the index name by passing the variable: FAISS.load_local(str(faiss_index_path), embeddings, index = Username) 
    retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=0.01, search_kwargs=dict(k=4))
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    # Load the RAG model
    system_prompt = system_prompt.format(image = memory.load_memory_variables({"prompt": question})["history"])

    response = llm.generate_text(system_prompt)
    print(response)
    return response