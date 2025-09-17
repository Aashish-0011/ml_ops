from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from torch.quantization import quantize_dynamic
from huggingface_hub import HfApi

# model_path = "launchco/eb3-llm-health/tree/main/eb3-health"
model_path = "launchco/eb3-llm-health"
quantized_local_dir = "eb3_quantized_local"     # where to save locally
repo_id = "aashish12/eb3_quantized"



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
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_enable_fp32_cpu_offload=True  # allow offload if VRAM is low
    # )

    # max_memory = {
    #     0: "14GiB",   # GPU0
    #     1: "14GiB",   # GPU1
    #     "cpu": "32GiB"
    # }

    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        subfolder="eb3-health",
        # quantization_config=bnb_config,
        device_map="auto",
        # max_memory=max_memory,
        # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
        dtype=torch.qint8
    )
    # return model
    return quantized_model

# Save quantized model locally
# -------------------
def save_quantized_model(model, tokenizer, save_dir=quantized_local_dir):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Quantized model saved at {save_dir}")


def upload_to_hf(local_dir=quantized_local_dir, repo_id=repo_id):
    api = HfApi()
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id
    )
    print(f"🚀 Model uploaded to https://huggingface.co/{repo_id}")


# # Quick test

# print('device-->>',  model.device)
# prompt = "Hello, how are you?"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_length=100)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
if __name__ == "__main__":
    print("🔥 Loading tokenizer...")
    tokenizer = tokenize_data()

    print("🔥 Loading and quantizing model...")
    model = load_model()

    print("💾 Saving quantized model locally...")
    save_quantized_model(model, tokenizer)

    print("☁️ Uploading to Hugging Face Hub...")
    upload_to_hf()
