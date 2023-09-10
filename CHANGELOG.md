# CHANGELOG

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
