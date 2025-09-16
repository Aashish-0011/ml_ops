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

    # 8-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True  # allow offload if VRAM is low
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        subfolder="eb3-health",
        quantization_config=bnb_config,
        device_map="auto",
        # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    return model

# # Quick test

# print('device-->>',  model.device)
# prompt = "Hello, how are you?"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_length=100)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
