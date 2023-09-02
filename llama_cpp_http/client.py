import json
import time
import random
from typing import Any, List, Mapping, Optional, Iterator

import requests
from websockets.sync.client import connect
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema.output import GenerationChunk
# from langchain.embeddings.base import Embeddings
# from langchain.pydantic_v1 import BaseModel


# class LlamaCppEmbeddingsClient(Embeddings, BaseModel):
#     endpoint: str | list[str] | tuple[str] = 'http://127.0.0.1:5000'
#     model: str = ''
#     n_predict: int = -1
#     ctx_size: int = 2048
#     batch_size: int = 512
#     temperature: float = 0.8
#     top_k: int = 40
#     top_p: float = 0.9
#     n_gpu_layers: int = 0
#     streaming: bool = True
#     verbose: bool = False
#
#     def _get_embedding(self) -> List[float]:
#         return list(np.random.normal(size=self.size))
#
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [self._get_embedding() for _ in texts]
#
#     def embed_query(self, text: str) -> List[float]:
#         return self._get_embedding()


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
        
        connected = False

        while not connected:
            try:
                with connect(url, open_timeout=10.0, close_timeout=10.0) as ws:
                    connected = True
                    ws.send(json.dumps(req))
                    
                    for msg in ws:
                        res = json.loads(msg)

                        if res['status'] == 'error':
                            ws.close()
                            raise Exception(res)
                        elif res['status'] == 'success':
                            ws.close()
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
            except TimeoutError as e:
                print('TimeoutError:', e)
                print('retrying')
                time.sleep(random.random() * 10)
                continue
            except Exception as e:
                print('Exception:', e)
                print('retrying')
                time.sleep(random.random() * 10)
                continue

