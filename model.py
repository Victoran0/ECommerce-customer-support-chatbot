from llama_cpp import Llama
import os

model_path = "llama-3.2-1B-it-Ecommerce-ChatBot-merged-F16.gguf"
n_threads = os.cpu_count()

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_batch=32,
    n_threads=n_threads,
    n_gpu_layers=-1,
    chat_format="llama-3"
)
