# GPU-Accelerated LLM Inference

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

## NVIDIA GPU Benchmark

The same DistilGPT2 model was tested on an NVIDIA Tesla T4 GPU using CUDA.

## Results

| Configuration | Tokens/second |
|---|---:|
| CPU baseline | 37.83 |
| CPU — 2 threads | 42.88 |
| NVIDIA Tesla T4 | **189.22** |

**Best measured GPU throughput: 189.22 tokens/sec**

**GPU speedup vs best CPU: ~4.4×**

In this experiment, the Tesla T4 achieved approximately 4.4× higher throughput than the best CPU configuration tested.

### GPU Benchmark Details

```text
GPU: NVIDIA Tesla T4
Generated tokens: 30
Average time: 0.1585 seconds
Tokens/second: 189.22
```

### Current Results
```bash
CPU baseline:       37.83 tokens/sec
Best CPU test:      42.88 tokens/sec
NVIDIA T4:         189.22 tokens/sec
```
This demonstrates the performance advantage of GPU acceleration for this LLM inference workload.

### Finding

**2 CPU threads gave the best result in this experiment: 42.88 tokens/second.**

This showed me that more CPU threads do not always mean faster LLM inference. Performance needs to be measured instead of assumed.

### Project Structure

```text
gpu-llm-inference/
│
├── benchmarks/
│   ├── baseline.txt
│   ├── first_profile.txt
│   ├── thread_results.txt
│   └── gpu_t4_results.txt
│
├── baseline.py
├── llm_profile.py
├── thread_benchmark.py
└── README.md
```

### What I Learned
- What LLM inference means
- How to run an LLM locally
- How to measure inference speed
- What a performance baseline is
- How CPU threads affect inference
- How to benchmark different configurations
- Why optimization should be based on measurements

### Next Steps

- Optimize NVIDIA GPU inference
- Test FP16/BF16
- Measure GPU memory usage
- Learn and apply CUDA optimization
- Test quantization
- Compare optimization techniques
- Build a final performance report

### Project Philosophy

**Measure → Find the bottleneck → Optimize → Measure again → Prove the improvement**

### Status

🟢 CPU baseline completed
🟢 CPU thread benchmarking completed
🔵 NVIDIA GPU/CUDA optimization — next stage
