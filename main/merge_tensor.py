from safetensors.torch import load_file, save_file

# Load all parts
part1 = load_file("./eb3_quantized_local/model-00001-of-00002.safetensors")
part2 = load_file("./eb3_quantized_local/model-00002-of-00002.safetensors")

# Merge
merged = {**part1, **part2}

# Save as single safetensors file
save_file(merged, "./eb3_quantized_local/model_merged.safetensors")
