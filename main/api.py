from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

from model_setup import model_path, model, tokenizer

print('model_path--->>>', model_path)

# Create API
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.get("/generate")
async def generate_text(prompt: str = "hi"):
    print('promt-->>', prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=100)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response_text}

@app.get("/")
async def root():
    return {"message": "EB3 Health Model API is running!"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)