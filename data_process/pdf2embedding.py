import psycopg2
import configparser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


config = configparser.ConfigParser()
config.read('config.ini')

db_name = config['database']['db_name']
host = config['database']['host']
password = config['database']['password']
port = config['database']['port']
user = config['database']['user']

conn = psycopg2.connect(
    dbname="postgres",
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")


loader = PyMuPDFReader()
documents = loader.load(file_path="data/NVIDIA-10Q-20242905.pdf")

text_parser = SentenceSplitter(chunk_size=1024)
text_chunks = []

doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="nvidia",
    embed_dim=384,  
)
vector_store.add(nodes)

