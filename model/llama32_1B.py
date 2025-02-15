from llama_index.llms.llama_cpp import LlamaCPP

model_url = "https://huggingface.co/Steven0090/Llama3.2-Instruct-1B-gguf/resolve/main/llama32_1B_q8_0.gguf"

llm = LlamaCPP(
    model_url=model_url,
    model_path=None,
    temperature=0.1,
    max_new_tokens=1024,
    context_window=4096,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)