from transformers import AutoModelForCausalLM
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import io
from typing import Optional
from pydantic import BaseModel
from main import embedding_storage, RAG_pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
model_id = "vikhyatk/moondream2"
revision = "2024-05-20"


class Item(BaseModel):
    llm: str
    embeddings: str
    question: str
    username: Optional[str] = "index"

llm = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the Moondream model
    llm["moondream_model"] = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision,
        torch_dtype=torch.float16
    ).to("cuda")
    # Load the embedding model from langchain
    llm["embedding_llm"] = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda'}
        )
    yield
    llm.clear()

app = FastAPI(lifespan=lifespan)

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    enc_image = llm["moondream_model"].encode_image(image)
    # print(enc_image.shape)
    description = llm["moondream_model"].answer_question(enc_image, "Describe this image.", tokenizer)
    embedding_storage(llm["embedding_llm"], description)

        # current directory
    directory_path = os.getcwd()
    directory_path = os.path.join(directory_path, "faiss_index")
    description_path = os.path.join(directory_path, "description.txt")
    with open(description_path, "a") as f:
        f.write(description + '\n')
    return {"description": description}

@app.post("/rag_pipeline")
async def rag_pipeline(item: Item):
    result = RAG_pipeline(item.llm, item.embeddings, item.question, item.username)
    return {"result": result}


import uvicorn

if __name__ == '__main__':
    # uvicorn.run("host:app", host='0.0.0.0', port=8000) # Change to this for actual usage
    uvicorn.run("host:app", host='0.0.0.0', port=8000, reload=True)