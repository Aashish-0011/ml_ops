# model_setup.py
# model_path = "../eb3-llm-health/eb3-health"
# https://huggingface.co/launchco/eb3-llm-health

model_path="/launchco/eb3-llm-health"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model (will handle sharded safetensors automatically)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # torch_dtype=torch.float16,   # or torch.float32 if you don’t have a GPU runtime
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


# Quick test
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))