import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


df = pd.read_parquet("data/products_clean.parquet")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma")
col = client.get_or_create_collection("products", metadata={"hnsw:space": "cosine"})

BATCH = 1000
for i in range(0, len(df), BATCH):
    b = df.iloc[i:i + BATCH]
    vecs = model.encode(b["search_text"].tolist(), normalize_embeddings=True).tolist()
    col.upsert(
        ids=b["product_id"].astype(str).tolist(),
        embeddings=vecs,
        documents=b["search_text"].tolist(),
        metadatas=[{"name": n, "category": c}
                   for n, c in zip(b["product_name"], b["category hierarchy"])],
    )
    print(f"{i + len(b)}/{len(df)}")