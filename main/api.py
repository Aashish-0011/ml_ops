from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
import json

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
    with the following keys:

    carrier_name,
    plan_name,
    plan_year,
    deductible_period,
    deductible_explanation,
    network_type,
    network_name,
    website,
    customer_service_phone,
    deductibles (with subkeys: in_network(single, family), out_of_network(single, family)),
    oop_maximums (with subkeys: in_network(single, family), out_of_network(single, family)),
    coinsurance (with subkeys: in_network, out_of_network),
    visits (with subkeys: pcp(in_network, out_of_network), specialist(in_network, out_of_network), urgent_care(in_network, out_of_network), emergency_room(in_network, out_of_network), preventive(in_network, out_of_network)),
    surgeries (with subkeys: outpatient(in_network, out_of_network), inpatient(in_network, out_of_network), newborn_delivery(in_network, out_of_network)),
    diagnostics (with subkeys: major(in_network, out_of_network)),
    prescriptions (with subkeys: rx_deductible(in_network, out_of_network), generic(in_network, out_of_network), brand(in_network, out_of_network), tier_3(in_network, out_of_network), tier_4(in_network, out_of_network), tier_5(in_network, out_of_network), mail_order(in_network, out_of_network)).

    Return only JSON.



    Input: {prompt}
    """
    print('promt-->>', ml_prompt)
    inputs = tokenizer(ml_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=1000)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print('response_text---->>', response_text)
    # return {"response": response_text}
     # Try to parse JSON safely
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError as e:
        print('excetio in  json decode-->>>',e )
        print('response_text',response_text )
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