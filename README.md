# python-llama-cpp-http

Python **llama.cpp** HTTP Server and **LangChain** LLM Client.

Assumption is that **GPU driver**, and **OpenCL** / **CUDA** libraries are installed.

Make sure you follow instructions of **llama.cpp** below for
OpenCL for AMDGPU/NVIDIA **CLBlast**,
or CUDA for NVIDIA **cuBLIS**
from https://github.com/ggerganov/llama.cpp .

Download https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q2_K.bin
to local directory `models`.

## Setup

### Arch/Manjaro

```bash
sudo pacman -Sy base-devel clang cmake ninja git python jq blas-openblas sqlite
yay -Sy clblast-git
```

### Ubuntu/Debian

```bash
sudo apt install build-essential python3-dev python3-venv libffi-dev libssl-dev clang cmake ninja-build git jq libopenblas-dev libsqlite-dev
```

### Required for CPU: Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp llama.cpp
cd llama.cpp
CC=clang CXX=clang++ make -j
```

### Optional for AMDGPU or NVIDIA: Build llama.cpp with CLBlast

```bash
git clone https://github.com/ggerganov/llama.cpp llama.cpp-clblast
cd llama.cpp-clblast
CC=clang CXX=clang++ make LLAMA_CLBLAST=1 -j
```

### Optional for NVIDIA: Build llama.cpp with cuBLIS

```bash
git clone https://github.com/ggerganov/llama.cpp llama.cpp-cublis
cd llama.cpp-cublis
CC=clang CXX=clang++ make LLAMA_CUBLAS=1 -j
```

### Setup python venv

```bash
python -m venv venv
source venv/bin/activate
python -m ensurepip --upgrade
pip install -U .
```

### Run
```bash
source venv/bin/activate
python -m llama_cpp_http.server --backend cpu --models-path ./models --llama-cpp-path ./llama.cpp
```

### Run Client Examples

Simple text completion Call `/api/1.0/text/completion`:

```bash
source venv/bin/activate
python -B misc/example_client_call.py | jq .
```

WebSocket stream `/api/1.0/text/completion`:

```bash
source venv/bin/activate
python -B misc/example_client_stream.py | jq -R '. as $line | try (fromjson) catch $line'
```

### Example Running Server

```bash
python -m llama_cpp_http.server --backend clblast --models-path ~/models/ --llama-cpp-path ~/llama.cpp-clblast --allow-cache-prompt true --cache-prompt-db ~/models/llama_cpp_http_cache_prompt.sqlite
```

```bash
gunicorn llama_cpp_http.server:get_app --bind 0.0.0.0:5000 --workers 1 --worker-class aiohttp.GunicornWebWorker --backend clblast --models-path ~/models/ --llama-cpp-path ~/llama.cpp-clblast --allow-cache-prompt true --cache-prompt-db ~/models/llama_cpp_http_cache_prompt.sqlite
```
