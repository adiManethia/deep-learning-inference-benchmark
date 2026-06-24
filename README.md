# Deep Learning Inference Benchmark Suite

A comprehensive, empirical performance evaluation framework comparing runtime engines, optimization tiers, and precision modes across distinct computer vision architectures on consumer-grade NVIDIA Ada Lovelace architecture (RTX 4050 Laptop GPU, Ubuntu 23.04/24.04, Python 3.12).

This project shifts focus away from model training accuracy onto **production engineering metrics**: throughput optimization, latency reduction, memory bandwidth utilization, graph compilation, and framework execution layers.

---

## 🚀 Key Engineering Objectives
* **Isolate Optimization Vectors:** Measure the performance delta between standard PyTorch Eager execution, PyTorch Graph Compilation (`torch.compile`), and language-agnostic ONNX Runtime graph execution.
* **Identify Hardware Bottlenecks:** Profile the latency penalties associated with host-to-device (CPU-to-GPU) memory copying across the PCIe bus when interfacing between native Python/NumPy structures and C++ acceleration runtimes.
* **Overcome Environment Constraints:** Solve real-world deployment challenges, including runtime dependency routing (`LD_LIBRARY_PATH`), dynamic shared library loading failures (`libcudnn`, `libcublas`), and execution fallback behavior.

---

## 🏗️ Core Architectures Evaluated
The benchmark profiles three distinct model topologies to assess how optimization mechanics scale with model depth, parameter density, and structural design:

1. **SmallCNN:** A lightweight custom convolutional architecture designed to isolate framework-level scheduling overhead from computational load.
2. **ResNet18:** A standard residual network featuring skip connections and batch normalization layers, serving as the industry baseline for standard workload execution.
3. **MobileNetV2:** An inverted-residual, depthwise-separable network optimized for parameter efficiency, serving as a test case for compiler graph-fusion capabilities.

---

## 📊 Empirical Benchmark Results

The table below summarizes the aggregated execution metrics collected across an active testing cycle at a stable evaluation **Batch Size of 32**:

| Model Topology | Execution Engine / Framework | Hardware Target | Mean Batch Latency (ms) | Throughput (Samples/Sec) |
| :--- | :--- | :--- | :--- | :--- |
| **SmallCNN** | PyTorch Eager Mode | CUDA (RTX 4050) | ~14.40 | ~2,222.22 |
| **SmallCNN** | ONNX Runtime | CUDA (RTX 4050) | 14.55 | 2,197.97 |
| **SmallCNN** | ONNX Runtime | CPU | 14.68 | 2,179.07 |
| **ResNet18** | PyTorch Compiled (`max-autotune`) | CUDA (RTX 4050) | ~26.33 | ~1,215.34 |
| **ResNet18** | PyTorch Eager Mode | CUDA (RTX 4050) | ~34.10 | ~938.41 |
| **ResNet18** | ONNX Runtime | CUDA (RTX 4050) | 301.79 | 106.03 |
| **ResNet18** | ONNX Runtime | CPU (Fallback) | 343.79 | 93.08 |
| **MobileNetV2** | PyTorch Eager Mode | CUDA (RTX 4050) | ~42.50 | ~752.94 |
| **MobileNetV2** | ONNX Runtime | CUDA (RTX 4050) | 95.56 | 334.85 |
| **MobileNetV2** | ONNX Runtime | CPU | 96.62 | 331.17 |

---

## 🧠 Core Technical Breakdowns & "Why We Did It"

### 1. PyTorch Eager vs. Compiled Mode (`torch.compile`)
* **What We Did:** We wrapped our native PyTorch model execution in `torch.compile(model, mode="max-autotune")`.
* **Why We Did It:** Standard PyTorch operates in *Eager Mode*, acting as a Python wrapper around C++ kernels. Every layer calls back into Python, introducing Python interpreter overhead. `torch.compile` intercepts this behavior, analyzing the execution sequence to perform **Operator Fusion** (e.g., melting Convolution + BatchNorm + ReLU into a single GPU instruction). 
* **The Engineering Takeaway:** On `ResNet18`, compilation successfully bypassed memory bandwidth constraints, rocketing throughput from ~938 samples/sec up to **~1,215 samples/sec**.

### 2. The ONNX Export Phase (The Hardware-Blind Blueprint)
* **What We Did:** Traced the live PyTorch computational execution using a dynamic tensor input and compiled it into a static Open Neural Network Exchange (`.onnx`) graph.
* **Why We Did It:** To remove dependency on PyTorch entirely. An ONNX file is a language-agnostic blueprint containing the raw mathematical matrices and layer definitions, completely decoupled from Python. It allows the model to be deployed into pure C++, Go, Rust, or embedded systems without requiring a heavy PyTorch installation.

### 3. The ONNX Runtime Bottleneck (NumPy & PCIe Overhead)
* **The Anomaly:** During testing, standard ONNX Runtime on CUDA underperformed significantly on heavier models (e.g., dropping to **106 samples/sec** on ResNet18).
* **The Root Cause:** ONNX Runtime is decoupled from PyTorch and consumes standard **NumPy arrays** residing in host memory (CPU RAM). In our basic loop, every inference request forced the machine to sync execution, copy the raw NumPy arrays over the motherboard's physical PCIe lanes to device memory (GPU VRAM), perform the math, and block execution to pass the result back to the host CPU. The physical travel time across the motherboard completely neutralized the GPU's raw processing advantage.
* **Production Fix:** In live enterprise environments, this is bypassed using **I/O Binding**, allocating fixed pointers directly in GPU memory using CUDA streams to keep data fully resident on the device.

---

## ⚖️ Numerical Precision Optimization: FP32 vs. FP16

While the baseline benchmarks were completed using standard single-precision floating-point formats, moving an architecture from production evaluation to enterprise scale requires choosing the correct numeric precision strategy:

### Full Precision (FP32)
* **Bit Allocation:** Uses 32 bits per number (1 sign bit, 8 exponent bits, 23 mantissa bits).
* **Characteristics:** Offers high numerical range and extreme accuracy. It is the default format used for model training to ensure weight gradients update smoothly without underflowing.
* **Deployment Drawback:** Consumes twice the memory footprint of half-precision formats, reducing cache locality and underutilizing specialized hardware execution units.

### Half Precision (FP16)
* **Bit Allocation:** Uses 16 bits per number (1 sign bit, 5 exponent bits, 10 mantissa bits).
* **Characteristics:** Cuts memory consumption exactly in half. This doubles effective memory bandwidth, reduces storage footprints, and lets your system store twice as many image activations inside the GPU cache.
* **Hardware Acceleration:** Activating FP16 unlocks the native **NVIDIA Tensor Cores** on your RTX 4050, which are silicon-level matrix math units engineered specifically to execute half-precision fused multiply-add (FMA) instructions simultaneously. 

---

## 🛠️ Infrastructure & System Debugging Ledger

Deploying deep learning systems requires navigating hardware-software alignment constraints. This project successfully resolved two major runtime library dependencies:

### 1. Resolving the Dynamic Linker/Loader Error (`libcudnn`)
* **Symptom:** `Failed to load library libonnxruntime_providers_tensorrt.so with error: libcudnn.so.8: cannot open shared object file`
* **Resolution:** Isolated the required runtime binaries inside the isolated Python virtual environment using pip-allocated wheel packages (`nvidia-cudnn-cu12`). We then modified the dynamic link loader path at the shell layer to map runtime dependencies directly to the compiled engine objects:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aditya/.venvs/torch/lib/python3.12/site-packages/nvidia/cudnn/lib