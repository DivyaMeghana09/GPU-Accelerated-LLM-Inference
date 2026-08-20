import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"
RUNS = 5

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

times = []

for i in range(RUNS):
    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30
        )

    end = time.perf_counter()

    elapsed = end - start
    times.append(elapsed)

    print(f"Run {i + 1}: {elapsed:.2f} seconds")

average_time = sum(times) / len(times)
tokens = output.shape[1] - inputs["input_ids"].shape[1]
tokens_per_second = tokens / average_time

print("\n--- Final Baseline ---")
print(f"Average time: {average_time:.2f} seconds")
print(f"Generated tokens: {tokens}")
print(f"Tokens/second: {tokens_per_second:.2f}")
print(f"Parameters: {model.num_parameters():,}")