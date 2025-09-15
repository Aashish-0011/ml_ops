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

    ml_prompt =  f"""
    Extract the insurance details from the following text and return them strictly in JSON format 
    with this structure:
    {{
      "carrier_name": "...",
      "plan_name": "...",
      "plan_year": "...",
      "deductible_period": "...",
      "deductible_explanation": "...",
      "network_type": "...",
      "network_name": "...",
      "website": "...",
      "customer_service_phone": "...",
      "deductibles": {{
        "in_network": {{"single": "...", "family": "..."}},
        "out_of_network": {{"single": "...", "family": "..."}}
      }},
      "oop_maximums": {{
        "in_network": {{"single": "...", "family": "..."}},
        "out_of_network": {{"single": "...", "family": "..."}}
      }},
      "coinsurance": {{
        "in_network": "...",
        "out_of_network": "..."
      }},
      "visits": {{
        "pcp": {{"in_network": "...", "out_of_network": "..."}},
        "specialist": {{"in_network": "...", "out_of_network": "..."}},
        "urgent_care": {{"in_network": "...", "out_of_network": "..."}},
        "emergency_room": {{"in_network": "...", "out_of_network": "..."}},
        "preventive": {{"in_network": "...", "out_of_network": "..."}}
      }},
      "surgeries": {{
        "outpatient": {{"in_network": "...", "out_of_network": "..."}},
        "inpatient": {{"in_network": "...", "out_of_network": "..."}},
        "newborn_delivery": {{"in_network": "...", "out_of_network": "..."}}
      }},
      "diagnostics": {{
        "major": {{"in_network": "...", "out_of_network": "..."}}
      }},
      "prescriptions": {{
        "rx_deductible": {{"in_network": "...", "out_of_network": "..."}},
        "generic": {{"in_network": "...", "out_of_network": "..."}},
        "brand": {{"in_network": "...", "out_of_network": "..."}},
        "tier_3": {{"in_network": "...", "out_of_network": "..."}},
        "tier_4": {{"in_network": "...", "out_of_network": "..."}},
        "tier_5": {{"in_network": "...", "out_of_network": "..."}},
        "mail_order": {{"in_network": "...", "out_of_network": "..."}}
      }}
    }}


    Input: {prompt}
    """
    print('promt-->>', ml_prompt)
    inputs = tokenizer(ml_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=1000)
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