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
from pony import orm

#
# env
#
parser = argparse.ArgumentParser(prog='server', description='Python llama.cpp HTTP Server')
parser.add_argument('--host', help='http server host', default='0.0.0.0')
parser.add_argument('--port', help='http server port', default=5000, type=int)
parser.add_argument('--timeout', help='llama.cpp timeout in seconds', default=300.0, type=float)
parser.add_argument('--backend', help='llama.cpp execution backend', default='cpu', type=str, choices=['cpu', 'clblast', 'cublis'])
parser.add_argument('--models-path', help='models directory path', default='~/models')
parser.add_argument('--llama-cpp-path', help='llama.cpp directory path', default='~/llama.cpp')
parser.add_argument('--allow-bg-cache-prompt', help='allow background prompting and caching for same prompt', type=bool, default=False)
parser.add_argument('--bg-db-prompt', help='database path for background caching of prompts', type=str, default='~/models/llama_cpp_bg_cache_prompt.sqlite')
parser.add_argument('--bg-n-tasks-per-prompt', help='number of background tasks for same prompt', type=int, default=3)
parser.add_argument('--bg-n-results-per-prompt', help='return last N results for same prompt', type=int, default=10)
parser.add_argument('--bg-llama-cpp-path', help='llama.cpp directory path used for background tasks', default='~/llama.cpp')
cli_args = parser.parse_args()

HOST = cli_args.host
PORT = cli_args.port
TIMEOUT = cli_args.timeout
BACKEND = cli_args.backend
MODELS_PATH = cli_args.models_path
LLAMA_CPP_PATH = cli_args.llama_cpp_path
ALLOW_BG_CACHE_PROMPT = cli_args.allow_bg_cache_prompt
BG_DB_PROMPT = cli_args.bg_db_prompt
BG_N_TASKS_PER_PROMPT = cli_args.bg_n_tasks_per_prompt
BG_N_RESULTS_PER_PROMPT = cli_args.bg_n_results_per_prompt
BG_LLAMA_CPP_PATH = cli_args.bg_llama_cpp_path
BG_LLAMA_CPP_PATH = cli_args.bg_llama_cpp_path

devices = []
devices_locks = []
task_queue = set()
bg_task_queue = asyncio.Queue()

def init_devices():
    global devices
    global devices_locks
    
    if BACKEND == 'cpu':
        # allow only ONE "device" concurrently
        n = (-1, -1, None, None)
        devices.append(n)
    elif BACKEND in ('clblast', 'cublis'):
        # number of devices depend on how many OpenCL finds
        # this also applies to NVIDIA cuBLIS / cuda devices
        for pi, p in enumerate(cl.get_platforms()):
            for di, d in enumerate(p.get_devices()):
                n = (pi, di, p, d)
                devices.append(n)
    else:
        raise ValueError(BACKEND)

    # create async looks, later used for async subprocesses
    for n in devices:
        devices_locks.append((n, asyncio.Lock()))

    print('devices_locks:')
    pprint(devices_locks)

#
# background prompting and caching
#
if ALLOW_BG_CACHE_PROMPT:
    db = orm.Database()

    class PromptOutputInfo(db.Entity):
        id = orm.PrimaryKey(str)
        prompt_hash = orm.Required(str, index=True)
        prompt = orm.Required(str)
        output_hash = orm.Required(str)
        output = orm.Required(str)
        info_hash = orm.Required(str)
        info = orm.Required(str)

    db.bind(provider='sqlite', filename='~/models/llama_cpp_http.sqlite', create_db=True)
    db.generate_mapping(create_tables=True)
    orm.set_sql_debug(True)


def build_llama_cpp_cmd(device, prompt, model, n_predict, ctx_size, batch_size, temperature, n_gpu_layers, top_k, top_p):
    pi, di, p, d = device
    shell_prompt = shlex.quote(prompt)
    cmd = []

    if BACKEND == 'cpu':
        pass
    elif BACKEND == 'clblast':
        cmd.extend([
            f'GGML_OPENCL_PLATFORM={pi}', 
            f'GGML_OPENCL_DEVICE={di}',
        ])
    elif BACKEND == 'clblis':
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
        '--prompt', shell_prompt,
    ])

    cmd = ' '.join(cmd)
    return cmd

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
        for dl in devices_locks:
            device, lock = dl
            pi, di, p, d = device

            if lock.locked():
                print(f'platform {pi}, device {di}: locked')
                continue
            
            print(f'platform {pi}, device {di}: available')
            
            cmd = build_llama_cpp_cmd(
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
        for dl in devices_locks:
            device, lock = dl
            pi, di, p, d = device

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

            cmd = build_llama_cpp_cmd(
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

    async with asyncio.TaskGroup() as tg:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)

                # run in background
                coro = _text_completion_stream(ws, id_, data)
                task = tg.create_task(coro)
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

    init_devices()
    app = get_app()
    web.run_app(app, host=HOST, port=PORT)
