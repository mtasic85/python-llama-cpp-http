import os
import json
import shlex
import asyncio
import argparse
from uuid import uuid4
from pprint import pprint

import pyopencl as cl
import aiohttp
from aiohttp import web
from aiohttp.web import middleware
from async_timeout import timeout

#
# env
#
HOST = os.getenv('HOST') or '0.0.0.0'
PORT = int(os.getenv('PORT') or 5000)
TIMEOUT = float(os.getenv('TIMEOUT') or 300.0)
PLATFORM = int(os.getenv('PLATFORM') or 0)
MODELS_PATH = os.getenv('MODELS_PATH', './models')
LLAMA_CPP_PATH = os.getenv('LLAMA_CPP_PATH', './llama.cpp')

parser = argparse.ArgumentParser(prog='server', description='Python llama.cpp HTTP Server')
parser.add_argument('--host', help='http server host', default=HOST)
parser.add_argument('--port', help='http server port', default=PORT, type=int)
parser.add_argument('--timeout', help='llama.cpp timeout in seconds', default=TIMEOUT, type=float)
parser.add_argument('--platform', help='pyopencl platform number', default=PLATFORM, type=int)
parser.add_argument('--models-path', help='models direcory path', default=MODELS_PATH)
parser.add_argument('--llama-cpp-path', help='llama.cpp direcory path', default=LLAMA_CPP_PATH)
cli_args = parser.parse_args()

HOST = cli_args.host
PORT = cli_args.port
TIMEOUT = cli_args.timeout
PLATFORM = cli_args.platform
MODELS_PATH = cli_args.models_path
LLAMA_CPP_PATH = cli_args.llama_cpp_path

cl_devices = []
cl_devices_locks = []
task_queue = set()

def init_cl_devices():
    global cl_devices
    global cl_devices_locks
    
    for pi, p in enumerate(cl.get_platforms()):
        for di, d in enumerate(p.get_devices()):
            n = (pi, di, p, d)
            cl_devices.append(n)

    for n in cl_devices:
        cl_devices_locks.append((n, asyncio.Lock()))

    print('cl_devices_locks:')
    pprint(cl_devices_locks)

#
# web server
#
routes = web.RouteTableDef()

@routes.post('/api/1.0/text/completion')
async def post_api_1_0_text_completion(request, id_):
    data = await request.json()
    # print('data:', data)

    model = data['model']
    prompt = data['prompt']
    n_predict = str(data.get('n_predict', '-1'))
    ctx_size = str(data.get('ctx_size', '2048'))
    batch_size = str(data.get('batch_size', '512'))
    temperature = str(data.get('temperature', '0.8'))
    top_k = str(data.get('top_k', '40'))
    top_p = str(data.get('top_p', '0.9'))
    n_gpu_layers = str(data.get('n_gpu_layers', '0'))
    
    shell_prompt = shlex.quote(prompt)
    # print('shell_prompt:', shell_prompt)
    
    proc = None
    stdout = None
    stderr = None

    while True:
        for (pi, di, p, d), lock in cl_devices_locks:
            if lock.locked():
                print(f'platform {pi}, device {di}: locked')
                continue
            
            print(f'platform {pi}, device {di}: available')

            cmd = ' '.join([
                f'GGML_OPENCL_PLATFORM={pi}', 
                f'GGML_OPENCL_DEVICE={di}', 
                f'{LLAMA_CPP_PATH}/main',
                '--model', f'{MODELS_PATH}/{model}',
                '--n-predict', n_predict,
                '--ctx-size', ctx_size,
                '--batch-size', batch_size,
                '--temp', temperature,
                '--n-gpu-layers', n_gpu_layers,
                '--top-k', top_k,
                '--top-p', top_p,
                '--mlock',
                '--no-mmap',
                '--simple-io',
                '--prompt', shell_prompt,
            ])
            # print('cmd:', cmd)

            async with lock:
                print(f'platform {pi}, device {di}: acquired')
                
                try:
                    async with timeout(TIMEOUT) as cm:
                        proc = await asyncio.create_subprocess_shell(
                            cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )

                        stdout, stderr = await proc.communicate()
                        stdout = stdout.decode('unicode-escape')
                        stderr = stderr.decode('unicode-escape')
                except asyncio.TimeoutError as e:
                    try:
                        proc.kill()
                    except Exception as e:
                        print('proc kill:', e)
                except Exception as e:
                    try:
                        proc.kill()
                    except Exception as e:
                        print('proc kill:', e)

            print(f'platform {pi}, device {di}: released')

            if cm.expired:
                res = {
                    'id': id_,
                    'status': 'error',
                    'error': 'timeout',
                    'task_queue_size': len(task_queue),
                }

                return web.json_response(res)

            break
        else:
            await asyncio.sleep(1.0)

        if stdout is not None or stderr is not None:
            break

        await asyncio.sleep(1.0)

    # post-process
    output = stdout[len(prompt):]
    info = stderr

    res = {
        'id': id_,
        'status': 'success',
        **data,
        'output': output,
        'info': info,
    }

    return web.json_response(res)


async def _text_completion_stream(ws, id_, data):
    # print('_text_completion_stream', ws, data)

    model = data['model']
    prompt = data['prompt']
    n_predict = str(data.get('n_predict', '-1'))
    ctx_size = str(data.get('ctx_size', '2048'))
    batch_size = str(data.get('batch_size', '512'))
    temperature = str(data.get('temperature', '0.8'))
    top_k = str(data.get('top_k', '40'))
    top_p = str(data.get('top_p', '0.9'))
    n_gpu_layers = str(data.get('n_gpu_layers', '0'))
    
    shell_prompt = shlex.quote(prompt)
    # print('shell_prompt:', shell_prompt)
    proc = None
    stdout = None
    stderr = None

    while True:
        for (pi, di, p, d), lock in cl_devices_locks:
            if lock.locked():
                print(f'platform {pi}, device {di}: locked')
                
                msg = {
                    'id': id_,
                    'status': 'info',
                    'task_queue_size': len(task_queue),
                }

                await ws.send_json(msg)
                await asyncio.sleep(1.0)
                continue
            
            print(f'platform {pi}, device {di}: available')

            cmd = ' '.join([
                f'GGML_OPENCL_PLATFORM={pi}', 
                f'GGML_OPENCL_DEVICE={di}', 
                f'{LLAMA_CPP_PATH}/main',
                '--model', f'{MODELS_PATH}/{model}',
                '--n-predict', n_predict,
                '--ctx-size', ctx_size,
                '--batch-size', batch_size,
                '--temp', temperature,
                '--n-gpu-layers', n_gpu_layers,
                '--top-k', top_k,
                '--top-p', top_p,
                '--simple-io',
                '--prompt', shell_prompt,
            ])
            # print('cmd:', cmd)

            async with lock:
                print(f'platform {pi}, device {di}: acquired')
                
                try:
                    async with timeout(TIMEOUT) as cm:
                        proc = await asyncio.create_subprocess_shell(
                            cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )

                        len_stdout = 0
                        stdout = []

                        while not proc.stdout.at_eof():
                            buf = await proc.stdout.read(128)
                            stdout.append(buf)
                            len_stdout += len(buf)
                            # print('buf:', buf)

                            if len_stdout >= len(shell_prompt):
                                chunk = buf.decode('unicode-escape')
                                
                                msg = {
                                    'id': id_,
                                    'status': 'chunk',
                                    'chunk': chunk,
                                    'done': False,
                                }

                                # print('msg:', msg)
                                await ws.send_json(msg)

                            await asyncio.sleep(0.01)

                        stdout = b''.join(stdout)
                        stdout = stdout.decode('unicode-escape')
                        
                        stderr = await proc.stderr.read()
                        stderr = stderr.decode('unicode-escape')
                except asyncio.TimeoutError as e:
                    try:
                        proc.kill()
                    except Exception as e:
                        print('proc kill:', e)
                except Exception as e:
                    try:
                        proc.kill()
                    except Exception as e:
                        print('proc kill:', e)

            print(f'platform {pi}, device {di}: released')

            if cm.expired:
                msg = {
                    'id': id_,
                    'status': 'error',
                    'error': 'timeout',
                    'task_queue_size': len(task_queue),
                }

                await ws.send_json(msg)
                await ws.close()

            break
        else:
            await asyncio.sleep(1.0)

        if stdout is not None or stderr is not None:
            break

        await asyncio.sleep(1.0)

    # post-process
    output = stdout[len(prompt):]
    info = stderr

    msg = {
        'id': id_,
        'status': 'success',
        **data,
        'output': output,
        'info': info,
        'done': True
    }

    await ws.send_json(msg)
    await ws.close()
    # print(msg)

@routes.get('/api/1.0/text/completion')
async def get_api_1_0_text_completion(request, id_):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print('websocket connection opened')

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)

            # run in background
            coro = _text_completion_stream(ws, id_, data)
            task = asyncio.create_task(coro)
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print(f'ws connection closed with exception {ws.exception()}')
            break
        else:
            print(f'ws msg.type:', msg.type)

    print('websocket connection closed')
    return ws

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

def get_app():
    app = web.Application(middlewares=[task_queue_middleware])
    app.add_routes(routes)
    return app

if __name__ == '__main__':
    print(cli_args)

    init_cl_devices()
    app = get_app()
    web.run_app(app, host=HOST, port=PORT)
