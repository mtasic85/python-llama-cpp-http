import json
import asyncio

import aiohttp

async def main():
    prompt = R'''The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Human: Lets chat about any topic I mention. You will find that they also tend to be more concise. Concise writing means using the fewest words possible to convey an idea clearly.
AI: Alright, I will answer any of your questions to be concise and clear.
Human: How are you?
AI:'''
    
    req = {
        'model': 'llama-2-7b-chat.ggmlv3.q2_K.bin',
        'prompt': prompt,
        'temperature': 0.75,
        # 'n_gpu_layers': 33,
        # 'stop': ['Human:']
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
