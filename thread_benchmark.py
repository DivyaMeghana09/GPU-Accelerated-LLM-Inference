import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"
THREADS = [1, 2, 4, 8]

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

for threads in THREADS:

    torch.set_num_threads(threads)

    # Warm-up
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=10
        )

    times = []

    for _ in range(3):
        start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30
            )

        end = time.perf_counter()
        times.append(end - start)

    average_time = sum(times) / len(times)
    tokens = output.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_second = tokens / average_time

    print(
        f"\nThreads: {threads}"
        f"\nAverage time: {average_time:.2f} seconds"
        f"\nTokens/second: {tokens_per_second:.2f}"
    )