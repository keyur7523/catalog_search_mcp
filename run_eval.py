import json, time, os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from hybrid import hybrid_search, rerank_search

queries = pd.read_csv("data/WANDS/dataset/query.csv", sep="\t")
labels = pd.read_csv("data/WANDS/dataset/label.csv", sep="\t")
exact = labels[labels["label"] == "Exact"].groupby("query_id")["product_id"].apply(lambda s: set(s.astype(str)))

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma").get_collection("products")

K = 10
hits, rr, n = 0, 0.0, 0
for _, row in queries.iterrows():
    relevant = exact.get(row["query_id"])
    if not relevant:
        continue
    #  res = col.query(query_embeddings=model.encode([row["query"]], normalize_embeddings=True).tolist(), n_results=K)
    # ids = res["ids"][0]
    # ids = hybrid_search(row["query"])
    ids = rerank_search(row["query"])
    n += 1
    if any(i in relevant for i in ids):
        hits += 1
    for rank, i in enumerate(ids, 1):
        if i in relevant:
            rr += 1 / rank
            break

summary = {"method": "hybrid_rerank", "queries": n, f"hit_rate@{K}": hits / n, f"mrr@{K}": rr / n}
print(summary)
os.makedirs("results", exist_ok=True)
# json.dump(summary, open(f"results/{int(time.time())}_vector_only.json", "w"), indent=2)
json.dump(summary, open(f"results/{int(time.time())}_{summary['method']}.json", "w"), indent=2)


