import os
import sys
import json
import time
import shlex
import asyncio

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

import argparse
import traceback
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
parser.add_argument('--platforms-devices', help='Custom platforms and devices indexes, example: 0:0,0:1', default="0:0")
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
devices: list[tuple[int, int, int]] = []
devices_locks: list[asyncio.Lock] = []
devices_wss: dict[tuple[int, int, int], Any] = {}
devices_procs: dict[tuple[int, int, int], asyncio.subprocess.Process] = {}
task_queue = set()

def init_devices():
    global devices
    global devices_locks
    
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

    print('devices_locks:')
    pprint(devices_locks)

def parse_llama_print_timings(text: str) -> dict:
    lines = [n for n in text.splitlines() if n.startswith('llama_print_timings:')]
    
    lines = [
        (
            '_'.join(line.split('=')[0].split(':')[1].split()),
            
            [n.strip() for n in line.split('=')[1].split('(')[0].split('/')] + 
            [n.strip() for n in line.split('=')[1].split('(')[1].replace(')', '').split(',')]
            if '/' in line else
            [n.strip() for n in line.split('=')[1].split('(')[0].split('/')]
        )
        for line in lines
    ]

    lines = dict(lines)

    for k, v in lines.items():
        lines[k] = {
            '_'.join(n.split()[1:]): float(n.split()[0])
            for n in v
        }

    return lines

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
        if index is not None and pi is not None and di is not None:
            cmd.extend([
                f'GGML_OPENCL_PLATFORM={pi}',
                f'GGML_OPENCL_DEVICE={di}',
            ])
    elif BACKEND == 'hipblas':
        pass
    elif BACKEND == 'cublas':
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
        '--no-mmap',
        '--simple-io',
        '--log-disable',
        '--prompt', shell_prompt,
    ])

    cmd = [str(n) for n in cmd]
    cmd = ' '.join(cmd)
    print('! cmd:', cmd)
    return cmd

def build_llama_cpp_embedding_cmd(device: tuple[int, int, int],
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
        if index is not None and pi is not None and di is not None:
            cmd.extend([
                f'GGML_OPENCL_PLATFORM={pi}',
                f'GGML_OPENCL_DEVICE={di}',
            ])
    elif BACKEND == 'hipblas':
        pass
    elif BACKEND == 'cublas':
        pass
    else:
        raise ValueError(BACKEND)

    cmd.extend([
        f'{LLAMA_CPP_PATH}/embedding',
        '--model', f'{MODELS_PATH}/{model}',
        '--n-predict', n_predict,
        '--ctx-size', ctx_size,
        '--batch-size', batch_size,
        '--temp', temperature,
        '--n-gpu-layers', n_gpu_layers,
        '--top-k', top_k,
        '--top-p', top_p,
        # '--mlock',
        '--no-mmap',
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
            # create new proc for model
            # t0: float = time.time()

            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            # map device to proc
            devices_procs[device] = proc

            stdout: bytes
            stderr: bytes

            if streaming:
                prev_buf: bytes
                buf: bytes
                text: str

                # receive original prompt in stdout
                # strip original prompt from return
                while not proc.stdout.at_eof():
                    # stdout
                    buf = await proc.stdout.read(1024)
                    stdout += buf

                    # skip original prompt
                    if len(stdout) > len(prompt_enc):
                        break

                    await asyncio.sleep(0.2)

                stdout = stdout[1 + len(prompt_enc):]
                stderr = b''
                
                # time-to-load model
                # print('time to load model:', time.time() - t0)

                # return left-overs from stdout as buf
                buf = stdout
                prev_buf = b''
                text = stdout.decode()
                print(f'[{index}, {pi}, {di}]:', repr(text))

                res = {
                    'id': id_,
                    'status': 'chunk',
                    'chunk': text,
                    'done': False,
                }

                yield False, res, None

                # read rest of tokens
                while not proc.stdout.at_eof():
                    buf = await proc.stdout.read(256)
                    prev_buf += buf
                    stdout += buf

                    try:
                        text = prev_buf.decode()
                    except Exception as e:
                        print('run_prompt buf.decode("unicode-escape") exception:', e)
                        continue

                    prev_buf = b''
                    print(f'[{index}, {pi}, {di}]:', repr(text))

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

                    await asyncio.sleep(0.2)
            else:
                stdout, stderr = await proc.communicate()
                stdout = stdout[1 + len(prompt_enc):]

            if streaming:
                if not stopped:
                    stderr = await proc.stderr.read()

            if stopped:
                print('stopword, trying to kill proc', proc.pid)

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
        print('timeout, trying to kill proc', proc.pid)

        try:
            proc.kill()
            await proc.wait()
            print('proc kill [timeout]')
        except Exception as e:
            print('proc kill [timeout]:', e)
        finally:
            proc = None

    print('!! stdout:')
    print(stdout)
    
    if not stdout:
        print('!! stderr:')
        print(stderr)

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

    # remove map device to proc
    if device in devices_procs:
        del devices_procs[device]

async def run_embedding(device: tuple[int, int, int],
                        id_: str,
                        prompt: str,
                        model: str,
                        n_predict: int,
                        ctx_size: int,
                        batch_size: int,
                        temperature: float,
                        n_gpu_layers: int,
                        top_k: int,
                        top_p: float) -> list[float] | None:
    index, pi, di = device
    stdout: bytes = b''
    stderr: bytes = b'' # b'llama.cpp ' + model.encode() # FIXME: default value is required?
    proc: asyncio.subprocess.Process | None = None
    vector: list[float] | None = None

    cmd: str = build_llama_cpp_embedding_cmd(
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
            # create new proc for model
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()
            stdout = stdout.decode().strip()
            stderr = stderr.decode().strip()
    except asyncio.TimeoutError as e:
        print('timeout, trying to kill proc', proc.pid)

        try:
            proc.kill()
            await proc.wait()
            print('proc kill [timeout]')
        except Exception as e:
            print('proc kill [timeout]:', e)
        finally:
            proc = None

    # print('!! stdout:')
    # print(stdout)

    # print('!! stderr:')
    # print(stderr)

    try:
        vector = [float(n) for n in stdout.split()]
    except Exception as e:
        print('run_embedding [error]:', e)

    return vector

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
            await asyncio.sleep(0.5)
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
    
    try:
        llama_timings = parse_llama_print_timings(stderr)
    except Exception as e:
        print('parse_llama_print_timings:', e)
        llama_timings = None
    
    if llama_timings:
        info = {
            **llama_timings
        }
    else:
        info = {
            'error': stderr,
        }

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
                await asyncio.sleep(0.5)
                continue

            print(f'index {index}, platform {pi}, device {di}: available')
            break
        else:
            await asyncio.sleep(0.5)
            continue

        if lock:
            break

    # map device to ws
    devices_wss[device] = ws

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

    try:
        llama_timings = parse_llama_print_timings(stderr)
    except Exception as e:
        print('parse_llama_print_timings:', e)
        llama_timings = None
    
    if llama_timings:
        info = {
            **llama_timings
        }
    else:
        info = {
            'error': stderr,
        }

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

    # remove map device to ws
    if device in devices_wss:
        del devices_wss[device]

@routes.get('/api/1.0/text/completion')
async def get_api_1_0_text_completion(request, id_: str):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print('websocket connection opened')

    try:
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
    except ExceptionGroup as e:
        traceback.print_exc()
        print('ExceptionGroup:', e)

        # find device
        device = [(k, v) for k, v in devices_wss.items() if v is ws][0][0]
        print('kill proc for device:', device)

        # kill proc
        while device not in devices_procs:
            await asyncio.sleep(0.5)

        proc = devices_procs[device]
        proc.kill()
        await proc.wait()
        del devices_procs[device]

        # close ws
        await ws.close()
        del devices_wss[device]

    print('websocket connection closed')
    return ws

@routes.post('/api/1.0/text/embeddings')
async def post_api_1_0_text_embeddings(request, id_: str):
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

    # no device required since it gets executed on CPU
    device = (None, None, None)

    output: list[float] | None = await run_embedding(
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
    )

    if output is None:
        info = {
            'error': 'could not embedding text into vector',
        }
    else:
        info = {}

    res = {
        'id': id_,
        'status': 'success',
        **data,
        'output': output,
        'info': info,
    }

    return web.json_response(res)

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
