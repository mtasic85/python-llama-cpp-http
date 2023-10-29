# llama.cpp

## Prerequisites

Determine your Linux distribution and install appropriate packages required for building [llama.cpp](https://github.com/ggerganov/llama.cpp).

### Arch/Manjaro

```bash
sudo pacman -Sy base-devel clang cmake ninja git python jq blas-openblas sqlite
yay -Sy clblast-git
```

### Debian/Ubuntu

```bash
sudo apt install build-essential python3-dev python3-venv python3-pip libffi-dev libssl-dev clang cmake ninja-build git jq libopenblas-dev libsqlite-dev
```

## CPU build - no acceleration

This solution is recommended for Apple and systems without AMD nor NVIDIA GPUs.
It is portable but slow. However, if you have large RAM (128GB) you will be able to run larger models that cannot fit in VRAM on your single GPU.

```bash
git clone https://github.com/ggerganov/llama.cpp llama.cpp
cd llama.cpp

# GCC
make -j

# or if you prefer clang
CC=clang CXX=clang++ make -j
```

## CLBlast for AMDGPU and NVIDIA - recommended

You will need to have installed appropriate GPU drivers with OpenCL support.

```bash
git clone https://github.com/ggerganov/llama.cpp llama.cpp-clblast
cd llama.cpp-clblast

# GCC
make LLAMA_CLBLAST=1 -j

# or if you prefer clang
CC=clang CXX=clang++ make LLAMA_CLBLAST=1 -j
```

## hipBLAS for AMDGPU using HIP/ROCm

```bash
git clone https://github.com/ggerganov/llama.cpp llama.cpp-hipblas
cd llama.cpp-hipblas

# gcc
make LLAMA_HIPBLAS=1 -j

# or if you prefer clang
CC=clang CXX=clang++ make LLAMA_HIPBLAS=1 -j
```

## cuBLAS for NVIDIA

```bash
git clone https://github.com/ggerganov/llama.cpp llama.cpp-cublas
cd llama.cpp-cublas

# gcc
make LLAMA_CUBLAS=1 -j

# or if you prefer clang
CC=clang CXX=clang++ make LLAMA_CUBLAS=1 -j
```

## Example Running Server based on OpenCL / CLBlast

1) Standalone python app:

```bash
python -m llama_cpp_http.server --backend clblast --models-path ~/models/ --llama-cpp-path ~/llama.cpp-clblast
```

2) Running using `gunicorn`:

```bash
gunicorn llama_cpp_http.server:get_app --bind 0.0.0.0:5000 --workers 1 --worker-class aiohttp.GunicornWebWorker --backend clblast --models-path ~/models/ --llama-cpp-path ~/llama.cpp-clblast
```
