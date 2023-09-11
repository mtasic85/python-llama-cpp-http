import os
import sys
import json
import shlex
import asyncio

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

import argparse
from typing import Any
from uuid import uuid4
from pprint import pprint
from random import choice
from hashlib import sha256
from collections.abc import AsyncIterator

import aiohttp
from aiohttp import web
from aiohttp.web import middleware
from async_timeout import timeout

#
# env
#
parser = argparse.ArgumentParser(prog='server', description='Python llama.cpp HTTP Server')
parser.add_argument('--host', help='http server host', default='0.0.0.0')
parser.add_argument('--port', help='http server port', default=5000, type=int)
parser.add_argument('--timeout', help='llama.cpp timeout in seconds', default=300.0, type=float)
parser.add_argument('--backend', help='llama.cpp execution backend', default='cpu', type=str, choices=['cpu', 'clblast', 'hipblas', 'cublas'])
parser.add_argument('--models-path', help='models directory path', default='models')
parser.add_argument('--llama-cpp-path', help='llama.cpp directory path', default='llama.cpp')
parser.add_argument('--platforms-devices', help='Custom platforms and devices indexes, example: 0:0,0:1', type=str, required=False)
cli_args = parser.parse_args()

HOST = cli_args.host
PORT = cli_args.port
TIMEOUT = cli_args.timeout
BACKEND = cli_args.backend
MODELS_PATH = cli_args.models_path
LLAMA_CPP_PATH = cli_args.llama_cpp_path
PLATFORMS_DEVICES = cli_args.platforms_devices

#
# devices
#
devices: list[(int, int, Any, Any)] = []
devices_locks: list[asyncio.Lock] = []
devices_procs: list[tuple[str | None, asyncio.subprocess.Process | None]] = []
task_queue = set()

def init_devices():
    global devices
    global devices_locks
    global devices_procs
    
    if BACKEND == 'cpu':
        # allow only ONE "device" concurrently
        n = (0, -1, -1)
        devices.append(n)
    elif BACKEND in ('clblast', 'hipblas', 'cublas'):
        # parse devices for example '0:0,0:1,1:0,1:1,2:0,2:1,2:2,2:3'
        pis_dis = PLATFORMS_DEVICES.split(',')

        for i, pi_di in enumerate(pis_dis):
            pi, di = pi_di.split(':')
            pi = int(pi)
            di = int(di)
            n = (i, pi, di)
            devices.append(n)
    else:
        raise ValueError(BACKEND)

    # create async looks, later used for async subprocesses
    for n in devices:
        devices_locks.append((n, asyncio.Lock()))

    # create devices_procs and set all values to None
    devices_procs = [(None, None)] * len(devices)

    print('devices_locks:')
    pprint(devices_locks)
    
    print('devices_procs:')
    pprint(devices_procs)

def build_llama_cpp_cmd(device: tuple[int, int, int],
                        prompt: str,
                        model: str,
                        n_predict: int,
                        ctx_size: int,
                        batch_size: int,
                        temperature: float,
                        n_gpu_layers: int,
                        top_k: int,
                        top_p: float) -> str:
    index, pi, di = device
    shell_prompt = shlex.quote(prompt)
    cmd = []

    if BACKEND == 'cpu':
        pass
    elif BACKEND == 'clblast':
        cmd.extend([
            f'GGML_OPENCL_PLATFORM={pi}',
            f'GGML_OPENCL_DEVICE={di}',
        ])
    elif BACKEND == 'hipblas':
        pass
    elif BACKEND == 'clblas':
        pass
    else:
        raise ValueError(BACKEND)

    cmd.extend([
        f'{LLAMA_CPP_PATH}/main',
        '--model', f'{MODELS_PATH}/{model}',
        '--n-predict', n_predict,
        '--ctx-size', ctx_size,
        '--batch-size', batch_size,
        '--temp', temperature,
        '--n-gpu-layers', n_gpu_layers,
        '--top-k', top_k,
        '--top-p', top_p,
        # '--mlock',
        # '--no-mmap',
        '--simple-io',
        '--log-disable',
        '--prompt', shell_prompt,
    ])

    cmd = [str(n) for n in cmd]
    cmd = ' '.join(cmd)
    print('! cmd:', cmd)
    return cmd

async def run_prompt(device: tuple[int, int, int],
                     id_: str,
                     prompt: str,
                     model: str,
                     n_predict: int,
                     ctx_size: int,
                     batch_size: int,
                     temperature: float,
                     n_gpu_layers: int,
                     top_k: int,
                     top_p: float,
                     stop: list[str] | None=None,
                     streaming: bool=False) -> AsyncIterator[(bool, str, str)]:
    index, pi, di = device
    stdout: bytes = b''
    stderr: bytes = b'llama.cpp ' + model.encode() # FIXME: default value is required?
    prompt_enc: bytes = prompt.encode()
    shell_prompt: str = shlex.quote(prompt)
    stop_enc = None if stop is None else [n.encode() for n in stop]
    stopped: bool = False
    read_stderr_task = None
    proc_model: str | None = None
    proc: asyncio.subprocess.Process | None = None

    print('? prompt:', repr(prompt))

    cmd: str = build_llama_cpp_cmd(
        device=device,
        prompt=prompt,
        model=model,
        n_predict=n_predict,
        ctx_size=ctx_size,
        batch_size=batch_size,
        temperature=temperature,
        n_gpu_layers=n_gpu_layers,
        top_k=top_k,
        top_p=top_p,
    )

    try:
        async with timeout(TIMEOUT) as cm:
            # get eager proc with its last used model
            proc_model, proc = devices_procs[index]

            if proc_model != model:
                if proc:
                    # close proc because model is wrong
                    proc.kill()
                    await proc.wait()
                    print('proc kill [wrong model]')

                # create new proc for model
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                devices_procs[index] = (model, proc)
                print('devices_procs:', devices_procs)

            if streaming:
                buf: bytes
                text: str

                # strip original prompt from return
                while not proc.stdout.at_eof():
                    # stdout
                    buf = await proc.stdout.read(16)
                    stdout += buf

                    # skip original prompt
                    if len(stdout) > len(prompt_enc):
                        break

                stdout = stdout[1 + len(prompt_enc):]

                # return left-overs from stdout as buf
                buf = stdout
                text = buf.decode('unicode-escape')

                res = {
                    'id': id_,
                    'status': 'chunk',
                    'chunk': text,
                    'done': False,
                }

                yield False, res, None

                # read rest of tokens
                while not proc.stdout.at_eof():
                    buf = await proc.stdout.read(128)
                    stdout += buf

                    # FIXME: requires better implementation
                    try:
                        text = buf.decode('unicode-escape')
                    except Exception as e:
                        print('run_prompt buf.decode("unicode-escape") exception:', e)
                        continue
                    
                    res = {
                        'id': id_,
                        'status': 'chunk',
                        'chunk': text,
                        'done': False,
                    }

                    yield False, res, None

                    # check for stop words
                    if stop_enc:
                        for n in stop_enc:
                            if n in stdout:
                                print('* stopped:', stop)
                                stdout = stdout[:stdout.index(n)]
                                stopped = True
                                break

                    if stopped:
                        break

                    await asyncio.sleep(0.1)
            else:
                stdout, stderr = await proc.communicate()
                stdout = stdout[1 + len(prompt_enc):]

            if stopped:
                try:
                    proc.kill()
                    await proc.wait()
                    print('proc kill [stop]')
                except Exception as e:
                    print('proc kill [stop]:', e)
                finally:
                    proc = None
            
            stdout = stdout.decode().strip()
            stderr = stderr.decode().strip()
    except asyncio.TimeoutError as e:
        try:
            proc.kill()
            await proc.wait()
            print('proc kill [timeout]')
        except Exception as e:
            print('proc kill [timeout]:', e)
        finally:
            proc = None

    print('!! stdout', repr(stdout))
    print('!! stderr', repr(stderr))

    # create eager proc for model
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    devices_procs[index] = (model, proc)
    print('devices_procs:', devices_procs)

    if cm.expired:
        res = {
            'id': id_,
            'status': 'error',
            'error': 'timeout',
            'task_queue_size': len(task_queue),
        }

        yield True, None, res
    else:
        yield True, stdout, stderr

#
# web server
#
routes = web.RouteTableDef()

@routes.post('/api/1.0/text/completion')
async def post_api_1_0_text_completion(request, id_: str):
    data = await request.json()
    
    model = data['model']
    prompt = data['prompt']
    n_predict = int(data.get('n_predict', '-1'))
    ctx_size = int(data.get('ctx_size', '2048'))
    batch_size = int(data.get('batch_size', '512'))
    temperature = float(data.get('temperature', '0.8'))
    top_k = int(data.get('top_k', '40'))
    top_p = float(data.get('top_p', '0.9'))
    n_gpu_layers = int(data.get('n_gpu_layers', '0'))
    stop = data.get('stop')
    
    # find avilable lock
    while True:
        for dl in devices_locks:
            device, lock = dl
            index, pi, di = device

            if lock.locked():
                print(f'index {index}, platform {pi}, device {di}: locked')
                continue
            
            print(f'index {index}, platform {pi}, device {di}: available')
            break
        else:
            await asyncio.sleep(1.0)
            continue

        if lock:
            break
    
    # run prompt
    stdout = None
    stderr = None

    async with lock:
        print(f'index {index}, platform {pi}, device {di}: acquired')
        
        g = run_prompt(
            device=device,
            id_=id_,
            prompt=prompt,
            model=model,
            n_predict=n_predict,
            ctx_size=ctx_size,
            batch_size=batch_size,
            temperature=temperature,
            n_gpu_layers=n_gpu_layers,
            top_k=top_k,
            top_p=top_p,
            stop=stop,
            streaming=False,
        )

        async for done, stdout, stderr in g:
            if done:
                break

    print(f'index {index}, platform {pi}, device {di}: released')

    # post-process
    output = stdout
    # info = stderr
    info = None

    res = {
        'id': id_,
        'status': 'success',
        **data,
        'output': output,
        'info': info,
    }

    return web.json_response(res)

async def _ws_text_completion_stream(ws, id_: str, data: dict):
    model = data['model']
    prompt = data['prompt']
    n_predict = int(data.get('n_predict', '-1'))
    ctx_size = int(data.get('ctx_size', '2048'))
    batch_size = int(data.get('batch_size', '512'))
    temperature = float(data.get('temperature', '0.8'))
    top_k = int(data.get('top_k', '40'))
    top_p = float(data.get('top_p', '0.9'))
    n_gpu_layers = int(data.get('n_gpu_layers', '0'))
    stop = data.get('stop')
    
    # find avilable lock
    while True:
        for dl in devices_locks:
            device, lock = dl
            index, pi, di = device

            if lock.locked():
                print(f'index {index}, platform {pi}, device {di}: locked')
                
                res = {
                    'id': id_,
                    'status': 'info',
                    'task_queue_size': len(task_queue),
                }

                await ws.send_json(res)
                await asyncio.sleep(1.0)
                continue

            print(f'index {index}, platform {pi}, device {di}: available')
            break
        else:
            await asyncio.sleep(1.0)
            continue

        if lock:
            break

    # run prompt
    stdout = None
    stderr = None

    async with lock:
        print(f'index {index}, platform {pi}, device {di}: acquired')
        
        g = run_prompt(
            device=device,
            id_=id_,
            prompt=prompt,
            model=model,
            n_predict=n_predict,
            ctx_size=ctx_size,
            batch_size=batch_size,
            temperature=temperature,
            n_gpu_layers=n_gpu_layers,
            top_k=top_k,
            top_p=top_p,
            stop=stop,
            streaming=True,
        )

        async for done, stdout_or_res, stderr_or_res in g:
            if done:
                if isinstance(stdout_or_res, dict) and stderr_or_res is None:
                    res = stdout_or_res
                    await ws.send_json(res)
                elif stdout_or_res is None and isinstance(stderr_or_res, dict):
                    res = stderr_or_res
                    await ws.send_json(res)
                    return
                
                if isinstance(stdout_or_res, str):
                    stdout = stdout_or_res

                if isinstance(stderr_or_res, str):
                    stderr = stderr_or_res

                break
            
            res = stdout_or_res
            await ws.send_json(res)

    print(f'index {device}, platform {pi}, device {di}: released')

    # post-process
    output = stdout
    # info = stderr
    info = None

    res = {
        'id': id_,
        'status': 'success',
        **data,
        'output': output,
        'info': info,
        'done': True
    }

    await ws.send_json(res)
    await ws.close()

@routes.get('/api/1.0/text/completion')
async def get_api_1_0_text_completion(request, id_: str):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print('websocket connection opened')

    async with asyncio.TaskGroup() as tg:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.PING:
                await ws.pong(msg.data)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)

                # run in background
                coro = _ws_text_completion_stream(ws, id_, data)
                task = tg.create_task(coro)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f'ws connection closed with exception {ws.exception()}')
                await ws.close()
                break
            else:
                print(f'ws msg.type:', msg.type)

    await task
    print('websocket connection closed')
    return ws

@routes.get('/')
async def get_index(request, id_: str):
    return web.Response(text='AI: Hello human.\nHuman: Hello AI.\n')

@middleware
async def task_queue_middleware(request, handler):
    global task_queue
    id_ = str(uuid4())
    task_queue.add(id_)

    try:
        resp = await handler(request, id_)
    except Exception as e:
        raise e
    finally:
        task_queue.remove(id_)

    return resp

async def get_app():
    app = web.Application(middlewares=[task_queue_middleware])
    app.add_routes(routes)
    return app

if __name__ == '__main__':
    print(cli_args)

    init_devices()
    app = asyncio.run(get_app())
    web.run_app(app, host=HOST, port=PORT)
