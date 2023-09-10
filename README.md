# python-llama-cpp-http

[![Build][build-image]]()
[![Status][status-image]][pypi-project-url]
[![Stable Version][stable-ver-image]][pypi-project-url]
[![Coverage][coverage-image]]()
[![Python][python-ver-image]][pypi-project-url]
[![License][mit-image]][mit-url]

## Installation

```bash
pip install llama_cpp_http
```

Python HTTP Server and [LangChain](https://python.langchain.com) LLM Client for [llama.cpp](https://github.com/ggerganov/llama.cpp).

Server has only two routes:
- **call**: for a prompt get whole text completion at once: `POST` `/api/1.0/text/completion`
- **stream**: for a prompt get text chunks via WebSocket: `GET` `/api/1.0/text/completion`

LangChain LLM Client has support for sync calls only based on Python packages `requirements` and `websockets`.

## Setup llama.cpp

Assumption is that **GPU driver**, and **OpenCL** / **CUDA** libraries are installed.

Make sure you follow instructions from `LLAMA_CPP.md` below for one of following:
- CPU - including Apple, recommended for beginners
- OpenCL for AMDGPU/NVIDIA **CLBlast**
- HIP/ROCm for AMDGPU **hipBLAS**,
- CUDA for NVIDIA **cuBLAS**

It is the easiest to start with just **CPU**-based version of [llama.cpp](https://github.com/ggerganov/llama.cpp) if you do not want to deal with GPU drivers and libraries.

### Install build packages

- Arch/Manjaro: `sudo pacman -Sy base-devel python git jq`
- Debian/Ubuntu: `sudo apt install build-essential python3-dev python3-venv python3-pip libffi-dev libssl-dev git jq`

### Clone repo

```bash
git clone https://github.com/mtasic85/python-llama-cpp-http.git
cd python-llama-cpp-http
```

Make sure you are inside cloned repo directory `python-llama-cpp-http`.

### Clone and compile llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp llama.cpp
cd llama.cpp
make -j
```

## Download Meta's Llama 2 7B model

Download GGUF model from https://huggingface.co/TheBloke/Llama-2-7B-GGUF to local directory `models`.

Our advice is to use model https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q2_K.gguf with minimum requirements, so it can fit in both RAM/VRAM.

### Setup python venv

```bash
python -m venv venv
source venv/bin/activate
python -m ensurepip --upgrade
pip install -U .
```

### Run Server

```bash
source venv/bin/activate
python -m llama_cpp_http.server --backend cpu --models-path ./models --llama-cpp-path ./llama.cpp
```

### Run Client Examples

1) Simple text completion call `/api/1.0/text/completion`:

```bash
source venv/bin/activate
python -B misc/example_client_call.py | jq .
```

2) WebSocket stream `/api/1.0/text/completion`:

```bash
source venv/bin/activate
python -B misc/example_client_stream.py | jq -R '. as $line | try (fromjson) catch $line'
```

## Licensing

**python-llama-cpp-http** is licensed under the MIT license. Check the LICENSE for details.