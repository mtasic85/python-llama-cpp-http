import json
import logging
from typing import Any, List, Mapping, Optional, Iterator

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema.output import GenerationChunk
from websockets.sync.client import connect

logger = logging.getLogger(__name__)


class LlamaCppClient(LLM):
    endpoint: str = 'http://127.0.0.1:5000'
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
        url = f'{self.endpoint}/api/1.0/text/completion'

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

            if self.verbose:
                logger.debug(msg['info'])

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

        url = f'{self.endpoint}/api/1.0/text/completion'

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
        }

        with connect(url, open_timeout=300, close_timeout=300) as ws:
            ws.send(json.dumps(req))
            
            for msg in ws:
                res = json.loads(msg)

                if res['status'] == 'error':
                    raise Exception(res)
                elif res['status'] == 'success':
                    break
                elif res['status'] != 'chunk':
                    continue

                chunk = GenerationChunk(
                    text=res['chunk'],
                    generation_info={'logprobs': logprobs},
                )

                yield chunk

                if run_manager:
                    run_manager.on_llm_new_token(
                        token=chunk.text,
                        verbose=self.verbose,
                        log_probs=logprobs,
                    )

        if self.verbose:
            logger.debug(res['info'])