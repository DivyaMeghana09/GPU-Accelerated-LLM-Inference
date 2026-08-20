### GPU-Accelerated LLM Inference

### Project Goal

Make Large Language Model (LLM) inference faster and measure the performance improvements.

### What I Built

I started with a small open-source LLM and tested how quickly it generates text on my CPU.

Then I tested different CPU thread settings to find the best performance.

### How It Works

```bash

User Prompt
     ↓
Tokenizer
     ↓
LLM
     ↓
CPU
     ↓
Generated Text
     ↓
Performance Measurement
```
### Baseline

Model: distilgpt2

Parameters: 81,912,576

Device: CPU

Baseline result:
```bash

Generated tokens: 30
Average time: 0.79 seconds
Tokens/second: 37.83
```
### CPU Thread Experiment

| Threads | Tokens/second |
| ------: | ------------: |
|       1 |         37.33 |
|       2 |     **42.88** |
|       4 |         34.75 |
|       8 |         35.06 |

### Finding

**2 CPU threads gave the best result in this experiment: 42.88 tokens/second.**

This showed me that more CPU threads do not always mean faster LLM inference. Performance needs to be measured instead of assumed.

### Project Structure

```bash

gpu-llm-inference/
│
├── benchmarks/
│   ├── baseline.txt
│   ├── first_profile.txt
│   └── thread_results.txt
│
├── baseline.py
├── llm_profile.py
├── thread_benchmark.py
└── README.md
```

### What I Learned
. What LLM inference means
. How to run an LLM locally
. How to measure inference speed
. What a performance baseline is
. How CPU threads affect inference
. How to benchmark different configurations
. Why optimization should be based on measurements

### Next Steps
. Test on an NVIDIA GPU
. Learn CUDA
. Compare CPU vs GPU inference
. Test FP16/BF16
. Test quantization
. Improve GPU memory usage
. Benchmark the optimized system

### Project Philosophy

**Measure → Find the bottleneck → Optimize → Measure again → Prove the improvement**

### Status

🟢 CPU baseline completed
🟢 CPU thread benchmarking completed
🔵 NVIDIA GPU/CUDA optimization — next stage