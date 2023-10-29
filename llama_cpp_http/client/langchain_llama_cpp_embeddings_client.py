__all__ = ['LlamaCppEmbeddingsClient']

from typing import List

import requests
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel


class LlamaCppEmbeddingsClient(Embeddings, BaseModel):
    endpoint: str | list[str] | tuple[str] = 'http://127.0.0.1:5000'
    model: str = ''
    n_predict: int = -1
    ctx_size: int = 2048
    batch_size: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    n_gpu_layers: int = 0
    verbose: bool = False

    def _get_embedding(self, prompt: str) -> List[float]:
        if isinstance(self.endpoint, (list, tuple)):
            endpoint = random.choice(self.endpoint)
        else:
            endpoint = self.endpoint

        url = f'{endpoint}/api/1.0/text/embeddings'

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

        r = requests.post(url, json=req)
        res = r.json()
        assert res['status'] == 'success'
        output: list[float] = res['output']
        return output

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)
