#
# NOTE: broken
#

import json
import random
import asyncio
# import logging
from typing import Any, List, Mapping, Optional, Iterator, AsyncIterator

import aiohttp
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema.output import GenerationChunk

# logger = logging.getLogger(__name__)

def iter_over_async(ait, loop):
    ait = ait.__aiter__()
    
    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None
    
    while True:
        done, obj = loop.run_until_complete(get_next())
        
        if done:
            break
        
        yield obj

class LlamaCppClient(LLM):
    endpoint: str | list[str] | tuple[str] = 'http://127.0.0.1:5000'
    model: str = ''
    n_predict: int = -1
    ctx_size: int = 2048
    batch_size: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    n_gpu_layers: int = 0
    streaming: bool = True
    verbose: bool = False

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            'endpoint': self.endpoint,
            'model': self.model,
            'n_predict': self.n_predict,
            'ctx_size': self.ctx_size,
            'batch_size': self.batch_size,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'n_gpu_layers': self.n_gpu_layers,
            'verbose': self.verbose,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'llama_cpp_client'

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call to llama_cpp_server endpoint."""

        if isinstance(self.endpoint, (list, tuple)):
            endpoint = random.choice(self.endpoint)
        else:
            endpoint = self.endpoint

        url = f'{endpoint}/api/1.0/text/completion'

        req = {
            'prompt': prompt,
            'model': self.model,
            'n_predict': self.n_predict,
            'ctx_size': self.ctx_size,
            'batch_size': self.batch_size,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'n_gpu_layers': self.n_gpu_layers,
            'stop': stop,
        }

        if self.streaming:
            combined_text_output = []
            
            for chunk in self._stream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
                combined_text_output.append(chunk.text)
            
            text = ''.join(combined_text_output)
        else:
            r = requests.post(url, json=req)
            res = r.json()
            assert res['status'] == 'success'
            text = res['output']

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        
        return text

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        print('aa')
        logprobs = None

        if isinstance(self.endpoint, (list, tuple)):
            endpoint = random.choice(self.endpoint)
        else:
            endpoint = self.endpoint

        url = f'{endpoint}/api/1.0/text/completion'

        req = {
            'prompt': prompt,
            'model': self.model,
            'n_predict': self.n_predict,
            'ctx_size': self.ctx_size,
            'batch_size': self.batch_size,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'n_gpu_layers': self.n_gpu_layers,
            'stop': stop,
        }

        timeout = aiohttp.ClientTimeout(total=90.0)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        res = json.loads(msg)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
                    
                    print('res:', res)

                    if res['status'] == 'error':
                        await ws.close()
                        raise Exception(res)
                    elif res['status'] == 'success':
                        await ws.close()
                        break
                    elif res['status'] != 'chunk':
                        continue

                    text = res['chunk']

                    chunk = GenerationChunk(
                        text=text,
                        generation_info={'logprobs': logprobs},
                    )

                    yield chunk

                    if run_manager:
                        run_manager.on_llm_new_token(
                            token=chunk.text,
                            verbose=self.verbose,
                            log_probs=logprobs,
                        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Yields results objects as they are generated in real time.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.
        """
        loop = asyncio.get_event_loop()
        agen = self._astream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs)
        
        for msg in iter_over_async(agen, loop):
            yield msg
