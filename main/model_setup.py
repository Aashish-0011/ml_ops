from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# model_path = "launchco/eb3-llm-health/tree/main/eb3-health"
model_path = "launchco/eb3-llm-health"



print(torch.cuda.is_available())

#  function to  tokenzi
def tokenize_data(model_path="launchco/eb3-llm-health"):
    # # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="eb3-health",
        trust_remote_code=True   # Required for custom models like EB3/Qwen2
    )

    return tokenizer

# print('tokenizr--->>>')
# Load model
def load_model(model_path="launchco/eb3-llm-health"):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,              # Enable 4-bit
        bnb_4bit_compute_dtype=torch.float16,  # Can try torch.bfloat16 if your GPU supports it
        bnb_4bit_use_double_quant=True, # Nested quantization for efficiency
        bnb_4bit_quant_type="nf4"       # NormalFloat4 (better than fp4 usually)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        subfolder="eb3-health",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    return model

# # Quick test

# print('device-->>',  model.device)
# prompt = "Hello, how are you?"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_length=100)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
