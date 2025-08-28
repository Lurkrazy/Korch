#!/usr/bin/env python3
"""
Debug script to demonstrate Korch's kernel selection and generation process.
This shows how Korch mixes Tensor Core and CUDA Core programs.

Usage: python debug_kernel_selection.py model.onnx device
Example: python debug_kernel_selection.py candy.onnx a100
"""

import sys
import os

# Add framework directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'framework'))

from calc import main
import argparse
from utils import sanitize_prepare_args

def demonstrate_kernel_selection():
    """
    Demonstrates Korch's kernel selection showing:
    1. Operator fission creating candidate kernels
    2. Profiling each kernel type (memory-bound vs compute-bound)
    3. ILP solver selecting optimal kernel combination
    4. Code generation creating mixed execution plan
    """
    
    print("=" * 80)
    print("🧠 KORCH KERNEL SELECTION AND GENERATION DEMONSTRATION")
    print("=" * 80)
    print("This demonstrates how Korch creates heterogeneous execution plans")
    print("that mix Tensor Core programs with CUDA Core programs.\n")
    
    print("📋 PROCESS OVERVIEW:")
    print("1. Operator Fission: Decompose model into primitive operations")
    print("2. Kernel Profiling: Profile each operation on different compute units")
    print("   • Memory-bound ops → TVM kernels (CUDA cores)")
    print("   • Conv ops → cuDNN (tensor cores available)")  
    print("   • GEMM ops → cuBLAS (tensor cores on A100)")
    print("3. Kernel Orchestration: ILP solver finds optimal execution plan")
    print("4. Code Generation: Create mixed execution CUDA code")
    print("\nLook for [KORCH DEBUG] messages to see the selection process!\n")
    
    # Parse arguments
    if len(sys.argv) != 3:
        print("Usage: python debug_kernel_selection.py <model.onnx> <device>")
        print("Example: python debug_kernel_selection.py candy.onnx a100")
        sys.exit(1)
    
    model_path = sys.argv[1]
    device = sys.argv[2]
    
    # Create mock arguments for main function
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("device", type=str) 
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--database_dir", type=str, default="./debug_database")
    parser.add_argument("--code_output_dir", type=str, default="./debug_output")
    
    args = parser.parse_args([model_path, device])
    args, config, target_info = sanitize_prepare_args(args)
    
    print(f"🎯 Running Korch on {model_path} for {device}")
    print(f"🗂️  Database: {args.database_dir}")  
    print(f"💾 Code output: {args.code_output_dir}")
    print("-" * 80)
    
    # Run the main Korch pipeline
    main(args, target_info, config)
    
    print("\n" + "=" * 80)
    print("✅ KERNEL SELECTION AND GENERATION COMPLETE!")
    print("=" * 80)
    print("📁 Check the generated files:")
    print(f"   • Candidate kernels: {args.database_dir}/subgraph*/candidate_kernel_graphs/")
    print(f"   • Optimized kernels: {args.database_dir}/subgraph*/lib_export/")
    print(f"   • Final mixed code: {args.code_output_dir}/subgraph*/forward_pass.cu")
    print("\nThe final CUDA code mixes:")
    print("🔧 TVM-generated kernels (CUDA cores) for memory-bound operations")
    print("🚀 cuDNN calls (tensor cores) for convolution operations") 
    print("⚡ cuBLAS calls (tensor cores on A100) for matrix operations")

if __name__ == "__main__":
    demonstrate_kernel_selection()