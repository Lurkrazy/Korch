# Korch Kernel Selection and Code Generation

This document explains where and how Korch selects between Tensor Core and CUDA Core programs, and generates mixed execution plans.

## 🎯 Key Components

### 1. Kernel Orchestration Solver (`framework/calc.py`)
**Location**: Lines 218-341  
The `KernelOrchestrationSolver` class implements the ILP (Integer Linear Programming) solver that selects the optimal combination of kernels for minimal end-to-end latency.

```python
# Main solver that creates heterogeneous execution plans
class KernelOrchestrationSolver:
    def solve(self) -> Solution:
        # Uses PuLP to solve binary programming problem
        # Selects optimal kernel combination minimizing latency
```

### 2. Kernel Profilers (`framework/profiler.py`)
**Location**: Lines 66-144  
Different profilers determine the best compute unit for each operation:

- **`MemBoundKernelProfiler`**: Uses TVM for memory-bound ops → **CUDA cores**
- **`ConvKernelProfiler`**: Uses cuDNN for convolutions → **Tensor cores available**  
- **`GemmKernelProfiler`**: Uses cuBLAS for matrix ops → **Tensor cores on A100**

### 3. Tensor Core Selection (`operators/gemm.h`)
**Location**: Lines 39-43  
The critical tensor core enablement code:

```cpp
// 🎯 KORCH TENSOR CORE SELECTION: Enable TF32 tensor cores on A100
if (tf32) {
    cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH);
    // This enables tensor cores for GEMM operations on A100 GPUs
    // Mixed precision: FP32 inputs/outputs, TF32 internal computation
}
```

### 4. Code Generation (`framework/codegen.py`)
**Location**: Lines 107-427  
Generates the final mixed execution CUDA code that chains:
- TVM kernels (CUDA cores) for memory-bound operations
- cuDNN calls (tensor cores) for convolutions
- cuBLAS calls (tensor cores on A100) for matrix multiplications

## 🔍 How to See Kernel Selection in Action

### Method 1: Use the Debug Script
```bash
python debug_kernel_selection.py model.onnx a100
```

This will show detailed output like:
```
🔧 [KORCH DEBUG] Using TVM (CUDA cores) for memory-bound kernel: 1
🚀 [KORCH DEBUG] Using cuDNN (Tensor Cores available) for Conv kernel: 2  
⚡ [KORCH DEBUG] Using cuBLAS (Tensor Cores) for GEMM kernel: 3

📋 [KORCH DEBUG] === EXECUTION PLAN ===
  Step 1: Kernel 1 → TVM (CUDA Cores) → 0.245 ms
  Step 2: Kernel 2 → cuDNN (Tensor Cores available) → 1.234 ms
  Step 3: Kernel 3 → cuBLAS (TF32 Tensor Cores) → 0.567 ms
🔀 [KORCH DEBUG] Heterogeneous execution mixing Tensor Cores + CUDA Cores
```

### Method 2: Examine Generated Code
After running Korch, check the generated files:
- **Candidate kernels**: `database_dir/subgraph*/candidate_kernel_graphs/`
- **Optimized kernels**: `database_dir/subgraph*/lib_export/`  
- **Final mixed code**: `code_output_dir/subgraph*/forward_pass.cu`

The final CUDA code will contain a mix of:
```cpp
// TVM-generated kernel launches (CUDA cores)
kernel_1<<<gridDim, blockDim>>>(params...);

// cuDNN calls (tensor cores available)  
cudnnConvolutionForward(cudnn, ...);

// cuBLAS calls (tensor cores on A100)
cublasGemmStridedBatchedEx(cublas, ...);
```

## 🧠 Understanding the Selection Logic

### Memory-Bound Operations → CUDA Cores
```python
# framework/profiler.py:88-97
def profile(self, candidate_kernel) -> float:
    if candidate_kernel.get_params() is not None:
        return INF  # Not memory-bound
    # Uses TVM to generate CUDA core kernels
    tvm_result = profile_main(...)
```

### Compute-Bound Operations → Tensor Cores (when available)

**Convolutions**:
```python  
# framework/profiler.py:104-119
# Uses cuDNN which internally decides on tensor cores
for i in range(8):
    params["algo"] = i
    cur_t = profile_conv(**params)  # cuDNN call
```

**Matrix Operations**:
```python
# framework/profiler.py:126-143  
use_tensor_cores = self.target_info.get_target_device_name() == "a100"
result = profile_gemm(..., use_tensor_cores)  # cuBLAS with tensor cores
```

## 🚀 Key Insight

Korch creates **heterogeneous execution plans** where:

1. **Operator Fission** decomposes models into primitive operations
2. **Kernel Orchestration** selects optimal compute units per operation:
   - Tensor cores for dense linear algebra (GEMM/Conv) 
   - CUDA cores for memory-bound operations
3. **Code Generation** creates dependency-aware execution graphs mixing both

This achieves up to **1.7x speedup** by using tensor cores where they excel while falling back to CUDA cores for operations that don't benefit from tensor cores.