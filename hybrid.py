import re
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

df = pd.read_parquet("data/products_clean.parquet")
ids = df["product_id"].astype(str).tolist()

def tok(t):
    return re.findall(r"[a-z0-9]+", t.lower())

print("Building BM25 index...")
bm25 = BM25Okapi([tok(t) for t in df["search_text"]])

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma").get_collection("products")

def vector_search(q, k=50):
    res = col.query(query_embeddings=model.encode([q], normalize_embeddings=True).tolist(), n_results=k)
    return res["ids"][0]

def bm25_search(q, k=50):
    scores = bm25.get_scores(tok(q))
    top = scores.argsort()[::-1][:k]
    return [ids[i] for i in top]

def hybrid_search(q, k=10, rrf_k=60):
    fused = {}
    for ranked in (vector_search(q), bm25_search(q)):
        for rank, pid in enumerate(ranked, 1):
            fused[pid] = fused.get(pid, 0) + 1 / (rrf_k + rank)
    return sorted(fused, key=fused.get, reverse=True)[:k]

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
text_by_id = dict(zip(ids, df["search_text"]))

def rerank_search(q, k=10, candidates=50):
    cands = hybrid_search(q, k=candidates)
    pairs = [(q, text_by_id[pid][:1000]) for pid in cands]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked[:k]]