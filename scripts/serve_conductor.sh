#!/bin/bash
# Serve the pinned reference model (§8b.9's recipe plus what this box needed).
#
# CUDA_HOME is a MERGED tree: nvcc ships in the venv wheel with a partial include tree, and the 151
# real headers ship in the system wheel with no nvcc. Neither is a usable CUDA_HOME alone.
#
# moe_backend=triton because the flashinfer CUTLASS path JIT-compiles sm_120 fused-MoE kernels with
# nvcc, and this box's split toolchain emits PTX 9.3 into a ptxas that accepts 9.0. Triton compiles
# its own kernels and needs none of that. This SELECTS A KERNEL rather than bypassing a correctness
# assertion — §8b.9 forbids CCCL_DISABLE_CTK_COMPATIBILITY_CHECK for exactly that reason, and this
# is not it.
CU=/root/cuda-home
export CUDA_HOME="$CU"
export PATH="$CU/bin:/root/vllm-env/bin:$PATH"
export LIBRARY_PATH="$CU/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CU/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0
exec /root/vllm-env/bin/vllm serve Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --served-model-name v3-conductor \
  --max-model-len 32768 --max-num-seqs 512 --enable-prefix-caching \
  --kernel-config '{"moe_backend":"triton","linear_backend":"triton","enable_jit_warmup":false,"enable_cutedsl_warmup":false}' \
  --port 8000 --host 127.0.0.1
