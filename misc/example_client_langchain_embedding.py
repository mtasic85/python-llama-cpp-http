import asyncio

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from llama_cpp_http.client import LlamaCppEmbeddingsClient

embeddings_model = LlamaCppEmbeddingsClient(
    endpoint='http://127.0.0.1:5000',
    model='llama-2-7b-chat.Q2_K.gguf',
    temperature=0.75,
    n_predict=512,
    ctx_size=2048,
    top_p=1,
    n_gpu_layers=0,
    verbose=True,
)

def main():
    embeddings = embeddings_model.embed_documents(
        [
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!"
        ]
    )

    print(len(embeddings))
    print(len(embeddings[0]))

if __name__ == '__main__':
    main()
