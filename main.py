import gradio as gr
from llama_index.core.query_engine import RetrieverQueryEngine
from model.llama32_1B import llm
from query_engine import retriever
from read_img import img_dict

query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

def query(prompt):
    response = query_engine.query(prompt)
    index_1 = int(response.source_nodes[0].metadata['source']) - 1
    index_2 = int(response.source_nodes[1].metadata['source']) - 1
    return response, img_dict[str(index_1)], img_dict[str(index_2)]

demo = gr.Interface(
    title="Using RAG for searching Nvidia Financial Report",
    fn=query,
    inputs=[gr.Text(label="Question")],
    outputs=[gr.Text(label="Answer"), gr.Image(label="Reference 1"), gr.Image(label="Reference 2")],
    flagging_options=[]
)

demo.launch(share=True)