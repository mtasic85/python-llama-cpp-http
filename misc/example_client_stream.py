import json
import asyncio

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

import aiohttp

async def main():
    prompt = (
        'The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.'
        'Human: I will provide you structured text.'
        'AI: Alright, I will parse it at the end as valid JSON.'
        'Human: You will parse text into valid JSON.'
        'AI: Alright, I will parse and output only valid JSON with "car_model": "string", "factory_discount": "string", "approx_price_saving": "string", "approx_factory_saving": "string", "offer_ends": "string".'
        'Human: 2023 Dodge Charger'
        '        Factory discount: $2,000-$3,000'
        '        Approximate price after savings: $32,000-$55,000'
        r'        Approximate factory savings: 4%-9%'
        '        Offer ends: June 19 or July 5 (depending on incentive)'
        'AI: That is an text to be parsed as valid JSON.'
        'Human: Parse text and output only valid JSON with fields `"car_model": "string", "factory_discount": "string", "approx_price_saving": "string", "approx_factory_saving": "string", "offer_ends": "string"`, and end conversation without explanation.'
        'AI:'
    )
    
    req = {
        'model': 'llama-2-7b-chat.Q2_K.gguf',
        # 'model': 'tinyllama-1.1b-intermediate-step-715k-1.5t.Q4_K_M.gguf',
        'n_predict': 512,
        'prompt': prompt,
        'temperature': 0.0,
        'n_gpu_layers': 35,
        'stop': ['Human:']
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect('http://127.0.0.1:5000/api/1.0/text/completion') as ws:
            await ws.send_json(req)

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    print(json.dumps(data))

                    if data.get('done'):
                        await ws.close()
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

if __name__ == '__main__':
    asyncio.run(main())
