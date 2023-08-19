# python-llama-cpp-http

Python llama.cpp HTTP Server and LangChain LLM Client

Assumption is that **GPU driver** and **OpenCL** library is installed.

Make sure you follow instructions of **llama.cpp** for **CLBlast** from https://github.com/ggerganov/llama.cpp .

Download https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q2_K.bin .

## Setup

### Arch/Manjaro

```bash
sudo pacman -Sy base-devel clang cmake ninja git python jq

# AUR
yay -S clblast-git
```

### Ubuntu/Debian

```bash
sudo apt install build-essential python3-dev python3-venv libffi-dev libssl-dev clang cmake ninja-build git jq
```

### Build llama.cpp with CLBlast

```bash
python -m venv ~/venv
source venv/bin/activate
python -m ensurepip --upgrade
pip install -r requirements.txt
```

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CLBLAST=1
cd ..
```

### Run
```bash
source venv/bin/activate
python -B server.py
```

### Run Client Examples

Simple text completion Call `/api/1.0/text/completion`:

```bash
source venv/bin/activate
python -B example_client_call.py | jq .
```

WebSocket stream `/api/1.0/text/completion`:

```bash
source venv/bin/activate
python -B example_client_stream.py | jq .
```