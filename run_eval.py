import argparse, json, time, os
import pandas as pd
from hybrid import vector_search, hybrid_search, rerank_search

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["vector", "hybrid", "rerank"], default="rerank")
p.add_argument("--candidates", type=int, default=50)
args = p.parse_args()

queries = pd.read_csv("data/WANDS/dataset/query.csv", sep="\t")
labels = pd.read_csv("data/WANDS/dataset/label.csv", sep="\t")
exact = labels[labels["label"] == "Exact"].groupby("query_id")["product_id"].apply(lambda s: set(s.astype(str)))

def search(q):
    if args.method == "vector":
        return vector_search(q, k=10)
    if args.method == "hybrid":
        return hybrid_search(q, k=10)
    return rerank_search(q, k=10, candidates=args.candidates)

K = 10
hits, rr, n = 0, 0.0, 0
for _, row in queries.iterrows():
    relevant = exact.get(row["query_id"])
    if not relevant:
        continue
    ids = search(row["query"])
    n += 1
    if any(i in relevant for i in ids):
        hits += 1
    for rank, i in enumerate(ids, 1):
        if i in relevant:
            rr += 1 / rank
            break

summary = {"method": args.method, "candidates": args.candidates if args.method == "rerank" else None,
           "queries": n, f"hit_rate@{K}": hits / n, f"mrr@{K}": rr / n}
print(summary)
os.makedirs("results", exist_ok=True)
json.dump(summary, open(f"results/{int(time.time())}_{args.method}.json", "w"), indent=2)