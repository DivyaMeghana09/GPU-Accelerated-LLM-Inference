from numba import cuda
import numpy as np
import time


@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)

    if i < a.size:
        c[i] = a[i] + b[i]


n = 10_000_000

a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)

# CPU benchmark
start = time.time()
c_cpu = a + b
cpu_time = time.time() - start

# GPU setup
c = np.zeros(n, dtype=np.float32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Warm-up
vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
cuda.synchronize()

# GPU benchmark
start = time.time()

vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
cuda.synchronize()

gpu_time = time.time() - start

print("=== CUDA Vector Addition Benchmark ===")
print("Elements:", n)
print("CPU time:", cpu_time, "seconds")
print("GPU kernel time:", gpu_time, "seconds")
print("Speedup:", cpu_time / gpu_time, "x")