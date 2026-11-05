import sys
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma").get_collection("products")

q = " ".join(sys.argv[1:])
res = col.query(
    query_embeddings=model.encode([q], normalize_embeddings=True).tolist(),
    n_results=10,
)

for pid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
    print(f"{pid:>6}  {1 - dist:.3f}  {meta['name']}  |  {meta['category']}")