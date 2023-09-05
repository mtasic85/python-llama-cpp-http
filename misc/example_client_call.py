import json
import asyncio

import aiohttp

async def main():
    prompt = R'''The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Human: I will provide you structured text.
AI: Alright, I will parse it at the end as valid JSON.
Human: You will parse text into valid JSON.
AI: Alright, I will parse and output only valid JSON with "car_model": "string", "factory_discount": "string", "approx_price_saving": "string", "approx_factory_saving": "string", "offer_ends": "string".
Human: 2023 Dodge Charger
        Factory discount: $2,000-$3,000
        Approximate price after savings: $32,000-$55,000
        Approximate factory savings: 4%-9%
        Offer ends: June 19 or July 5 (depending on incentive)
AI: That is an text to be parsed as valid JSON.
Human: Parse text and output only valid JSON with fields `"car_model": "string", "factory_discount": "string", "approx_price_saving": "string", "approx_factory_saving": "string", "offer_ends": "string"`, and end conversation without explanation.
AI:'''
    
    req = {
        # 'model': 'llama-2-7b-chat.ggmlv3.q2_K.bin',
        'model': 'llama-2-7b-chat.Q3_K_M.gguf',
        'prompt': prompt,
        # 'temperature': 0.75,
        # 'n_gpu_layers': 33,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post('http://127.0.0.1:5000/api/1.0/text/completion', json=req) as resp:
            data = await resp.json()
            print(json.dumps(data))

if __name__ == '__main__':
    asyncio.run(main())
