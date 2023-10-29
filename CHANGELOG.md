# CHANGELOG

## v0.3.0

Added:
    - client/langchain_llama_cpp_client.py
    - client/langchain_llama_cpp_embeddings_client.py
    - misc/example_client_langchain_embedding.py

Changed:
    - llama.cpp instructions
    - `client.py` is not `client` package
    - renamed misc/example_client_call_2.py to example_client_call_react.py

Fixed:
    - misc/example_client_langchain_stream.py

Removed:
    - misc/example_client_stream_codellama.py

## v0.2.13
## v0.2.12
## v0.2.11
## v0.2.10
## v0.2.9
## v0.2.8
## v0.2.7

Fixed:
    - Parsing subprocess output

## v0.2.6

Changed:
    - Forced immediate kill of subprocess if websocket is closed.

## v0.2.5

Changed:
    - Forced immediate kill of subprocess. In that case info/stderr is not returned.

## v0.2.4

Added:
    - llama timing info.

Changed:
    - Default --platforms-devices "0:0"

Fixed:
    - Token unicode string decoding.

## v0.2.3

Changed:
    - Do not eagerly load models.

## v0.2.2

Fixed:
    - Do not import server on package import.

## v0.2.1

Changed:
    - Eager/optimistic model loading.
    - Disable llama.cpp log traces.

## v0.2.0

Added:
    - Using `uvloop` speed interaction with `llama.cpp` ~2x.

Changed:
    - Does not return `info` field in responses as `stderr` from `llama.cpp`.

Removed:
    - `pyopencl` usage at in code.
    - Caching using `PonyORM` and `sqlite3`.

## v0.1.1

Added:
    - Homepage URL.
    - Repository URL.

Removed:
    - Removed `pyopencl` package from requirements.

## v0.1.0

Added:
    - HTTP Server based on `aiohttp`, and deployed using `gunicorn`.
    - HTTP Client for LangChain based on `websockets` for sync calls.
    - `pyopencl` is used to determin available **OpenCL** devices (GPUs).
    - Instructions in `LLAMA_CPP.md` how to build [llama.cpp](https://github.com/ggerganov/llama.cpp) w/o accelaration.
    - Cahcing using `PonyORM` and `sqlite3`.
