import json
import asyncio

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

import aiohttp

async def main():
    prompt = 'the water is freezing today'
    
    req = {
        'model': 'llama-2-7b-chat.Q2_K.gguf',
        'n_predict': 512,
        'prompt': prompt,
        'temperature': 0.75,
        'n_gpu_layers': 0,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post('http://127.0.0.1:5000/api/1.0/text/embeddings', json=req) as resp:
            data = await resp.json()
            print(json.dumps(data))

if __name__ == '__main__':
    asyncio.run(main())
