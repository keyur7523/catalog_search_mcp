import time, statistics, sys
from hybrid import model, col, bm25, ids, tok, text_by_id, reranker

def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000

queries = [
    "gray dining set for 4",
    "mid century walnut nightstand",
    "outdoor bench that won't rust",
    "queen platform bed solid wood",
    "modern black desk with drawers",
]

print("Warming up...", file=sys.stderr)
_ = model.encode(["warmup"], normalize_embeddings=True)
_ = reranker.predict([("warmup", "warmup")])

rows = []
for q in queries:
    emb, t_emb = timed(lambda: model.encode([q], normalize_embeddings=True).tolist())
    v_ids, t_vec = timed(lambda: col.query(query_embeddings=emb, n_results=50)["ids"][0])
    b_ids, t_bm = timed(lambda: [ids[i] for i in bm25.get_scores(tok(q)).argsort()[::-1][:50]])
    fused = {}
    for ranked in (v_ids, b_ids):
        for rank, pid in enumerate(ranked, 1):
            fused[pid] = fused.get(pid, 0) + 1 / (60 + rank)
    cands = sorted(fused, key=fused.get, reverse=True)[:50]
    _, t_rr = timed(lambda: reranker.predict([(q, text_by_id[pid][:1000]) for pid in cands]))
    total = t_emb + t_vec + t_bm + t_rr
    rows.append({"embed_ms": t_emb, "vector_ms": t_vec, "bm25_ms": t_bm, "rerank_ms": t_rr, "total_ms": total})
    print(f"{q:40}  emb={t_emb:6.0f}  vec={t_vec:6.0f}  bm25={t_bm:6.0f}  rerank={t_rr:6.0f}  total={total:6.0f}")

print("\nAverages (ms):")
for k in ["embed_ms", "vector_ms", "bm25_ms", "rerank_ms", "total_ms"]:
    print(f"  {k}: {statistics.mean(r[k] for r in rows):.0f}")