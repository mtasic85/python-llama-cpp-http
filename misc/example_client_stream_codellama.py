import json
import asyncio

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

import aiohttp

async def main():
    # prompt = R'''### Instruction: You will get text of an instruction. You will parse it as valid JSON using schema ```{"content": "string"}```. You will parse JSON in a single message. Value of parsed "content" field can be either "file_upload" or "table_view". Here are two examples ```{"content": "file_upload"}``` and ```{"content": "table_view"}```. Parse instruction "Upload files" as valid JSON.
    # prompt = R'''### Instruction: You will parse instruction as valid JSON using schema ```{"content": "string"}```. Value of parsed "content" field can be either "file_upload" or "table_view". Here are two examples ```{"content": "file_upload"}``` and ```{"content": "table_view"}```. Parse instruction "Upload files" as valid JSON.
    # prompt = R'''### Instruction: You will parse instruction as valid JSON using schema ```{"content": "string"}```. Value of parsed "content" field can be either "file_upload" or "table_view". Parse instruction "Upload files" as valid JSON.
#     prompt = R'''### Instruction: You will parse instruction as valid JSON using schema {"content": "string"}. Value of parsed "content" field can be either "file_upload" or "table_view". Parse instruction "Upload files" as valid JSON.
# ### Response:'''
    
#     prompt = R'''The following is a technical conversation between Human and AI. Conversation is the shortest possible.
# Human: You will get text of an instruction. You will parse it as valid JSON using schema ```{"content": "string"}```. You will parse JSON in a single message. Value of parsed "content" field can be either "file_upload" or "table_view". Here are two examples ```{"content": "file_upload"}``` and ```{"content": "table_view"}```.
# AI: Alright, that is JSON schema that I will use.
# Human: I will now provide single instruction.
# AI: Alright, I will parse it as valid JSON, output will be just valid JSON format, and end conversation immediately.
# Human: Parse instruction "Upload files" as valid JSON, and end conversation immediately after that without further follow up messages and explanations.
# AI:'''

    prompt = R'''The following is a technical conversation between Human and AI. Conversation is the shortest possible. AI does not explain reasoning, and it is not talkative.
Human: You will get text of an instruction. You will parse it as valid JSON using schema ```{"content": "string"}```. You will parse JSON in a single message. Value of parsed "content" field can be either "file_upload" or "table_view". Here are two examples ```{"content": "file_upload"}``` and ```{"content": "table_view"}```.
AI: Alright, that is JSON schema that I will use.
Human: I will now provide single instruction.
AI: Alright, I will parse it as valid JSON, output will be just valid JSON format, and end conversation immediately.
Human: Parse instruction "Upload files" as valid JSON, and end conversation immediately after that without further follow up messages and explanations.
AI:'''
    
    req = {
        'model': 'llama-2-7b-chat.Q2_K.gguf',
        'prompt': prompt,
        'temperature': 0.0,
        'n_gpu_layers': 32,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect('http://127.0.0.1:5000/api/1.0/text/completion') as ws:
            await ws.send_json(req)

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    # print(json.dumps(data))

                    if 'chunk' in data:
                        print(data['chunk'], end='')

                    if data.get('done'):
                        await ws.close()
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

if __name__ == '__main__':
    asyncio.run(main())
