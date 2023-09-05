from langchain.callbacks import StdOutCallbackHandler
from llama_cpp_http.client import LlamaCppClient

llm = LlamaCppClient(
    endpoint='ws://127.0.0.1:5000',
    # model='llama-2-7b-chat.ggmlv3.q2_K.bin',
    model='llama-2-7b-chat.Q3_K_M.gguf',
    temperature=0.75,
    n_predict=512,
    ctx_size=2048,
    top_p=1,
    n_gpu_layers=33,
    streaming=True,
    verbose=True,
)

def demo1():
    stdout_cb = StdOutCallbackHandler()

    prompt = R'''The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Human: Lets chat about any topic I mention. You will find that they also tend to be more concise. Concise writing means using the fewest words possible to convey an idea clearly.
AI: Alright, I will answer any of your questions to be concise and clear.
Human: How are you?
AI:'''

    output = llm(
        prompt=prompt,
        callbacks=[stdout_cb],
    )

    print(output)

if __name__ == '__main__':
    demo1()
