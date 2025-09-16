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

# @app.get("/generate")
# async def generate_text(prompt: str = "hi"):
#     ml_prompt = f"""
#     Extract the insurance details from the following text and return them strictly in JSON format 
#     with the following keys:

#     carrier_name,
#     plan_name,
#     plan_year,
#     deductible_period,
#     deductible_explanation,
#     network_type,
#     network_name,
#     website,
#     customer_service_phone,
#     deductibles (with subkeys: in_network(single, family), out_of_network(single, family)),
#     oop_maximums (with subkeys: in_network(single, family), out_of_network(single, family)),
#     coinsurance (with subkeys: in_network, out_of_network),
#     visits (with subkeys: pcp(in_network, out_of_network), specialist(in_network, out_of_network), urgent_care(in_network, out_of_network), emergency_room(in_network, out_of_network), preventive(in_network, out_of_network)),
#     surgeries (with subkeys: outpatient(in_network, out_of_network), inpatient(in_network, out_of_network), newborn_delivery(in_network, out_of_network)),
#     diagnostics (with subkeys: major(in_network, out_of_network)),
#     prescriptions (with subkeys: rx_deductible(in_network, out_of_network), generic(in_network, out_of_network), brand(in_network, out_of_network), tier_3(in_network, out_of_network), tier_4(in_network, out_of_network), tier_5(in_network, out_of_network), mail_order(in_network, out_of_network)).

#     Return only JSON.

#     Input: {prompt}
#     """

#     print("prompt-->>", ml_prompt)

#     inputs = tokenizer(ml_prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=1200,   # safer than max_length
#         temperature=0.2,       # keep responses more deterministic
#         do_sample=False        # avoid random junk
#     )
#     response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     print("raw response---->>", response_text)

#     # Try to extract JSON from model response
#     cleaned = response_text.strip()
#     # remove code fences if they exist
#     cleaned = cleaned.replace("```json", "").replace("```", "").strip()

#     # Sometimes the model echoes the instruction, strip leading junk
#     json_start = cleaned.find("{")
#     json_end = cleaned.rfind("}")
#     if json_start != -1 and json_end != -1:
#         cleaned = cleaned[json_start:json_end + 1]

#     try:
#         response_json = json.loads(cleaned)
#     except json.JSONDecodeError as e:
#         print("JSON decode error:", e)
#         response_json = {"raw_output": response_text}

#     return {"insurance_details": response_json}


@app.get("/generate")
async def generate_text(prompt: str = "hi"):

    # prompt = "Hello, how are you?"
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # outputs = model.generate(**inputs, max_length=100)

    # ml_prompt=f"Format the following document text into EB3 format:\n\ndocument:\n\n{prompt}"
    # 

    # print("prompt-->>", ml_prompt)
    print('start')
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate( **inputs, max_length=2200)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    

    print('done')

    print("\n\n\n\nraw response---->>", response_text)

    return {"insurance_details": response_text}


@app.get("/")
async def root():
    return {"message": "EB3 Health Model API is running!"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)