import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(4)
MODEL_NAME = "distilgpt2"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# Warm-up
with torch.no_grad():
    model.generate(
        **inputs,
        max_new_tokens=10
    )

print("Starting profiling...")

start = time.perf_counter()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=30
    )

end = time.perf_counter()

elapsed = end - start
tokens = output.shape[1] - inputs["input_ids"].shape[1]

print("\n--- Profiling Result ---")
print(f"Total time: {elapsed:.2f} seconds")
print(f"Generated tokens: {tokens}")
print(f"Tokens/second: {tokens / elapsed:.2f}")