from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

from model_setup import model_path, tokenize_data, load_model

tokenizer=tokenize_data()
model=load_model()

print('model_path--->>>', model_path)

# Create API
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.get("/generate")
async def generate_text(prompt: str = "hi"):

    ml_prompt = f"""
    Extract the insurance details from the following text and return them strictly in JSON format with keys:
    ["insurance_name", "policy_number", "expiry_date", "coverage_amount"]

    Input: {prompt}
    """
    print('promt-->>', ml_prompt)
    inputs = tokenizer(ml_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=100)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return {"response": response_text}
     # Try to parse JSON safely
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: attempt cleanup (strip code fences, extra text)
        cleaned = response_text.strip().replace("```json", "").replace("```", "")
        try:
            response_json = json.loads(cleaned)
        except:
            response_json = {"raw_output": response_text}

    return {"insurance_details": response_json}


@app.get("/")
async def root():
    return {"message": "EB3 Health Model API is running!"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)