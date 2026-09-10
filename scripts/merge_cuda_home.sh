#!/bin/bash
# nvcc ships in the venv wheel with a partial include tree; the 151 real headers (cuda_runtime.h
# among them) ship in the SYSTEM dist-packages wheel with no nvcc. Neither is a usable CUDA_HOME
# alone, so merge them by symlink into one. Symlinks, not copies: nothing is duplicated and the
# merge is reversible by deleting one directory.
set -e
V=/root/vllm-env/lib/python3.12/site-packages/nvidia/cu13
S=/usr/local/lib/python3.12/dist-packages/nvidia/cu13
H=/root/cuda-home
rm -rf "$H"; mkdir -p "$H/include" "$H/lib"
ln -sfn "$V/bin" "$H/bin"
# venv first (nvcc's own crt/nvvm/cccl), then system with -n so it never clobbers them.
cp -rsn "$V"/include/* "$H/include/" 2>/dev/null || true
cp -rsn "$S"/include/* "$H/include/" 2>/dev/null || true
cp -rsn "$V"/lib/* "$H/lib/" 2>/dev/null || true
cp -rsn "$S"/lib/* "$H/lib/" 2>/dev/null || true
# ld wants unversioned .so names; the wheels ship only libfoo.so.13 (§8b.9 failure mode 5).
for f in "$H"/lib/*.so.*; do
  [ -e "$f" ] || continue
  base=$(basename "$f"); stem=${base%%.so.*}
  [ -e "$H/lib/$stem.so" ] || ln -sf "$f" "$H/lib/$stem.so"
done
echo "nvcc:            $([ -x $H/bin/nvcc ] && echo yes || echo NO)"
echo "cuda_runtime.h:  $([ -f $H/include/cuda_runtime.h ] && echo yes || echo NO)"
echo "headers:         $(ls $H/include | wc -l)"
echo "libs:            $(ls $H/lib | wc -l)"
