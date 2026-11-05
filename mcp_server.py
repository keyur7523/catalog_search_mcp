from mcp.server.fastmcp import FastMCP
from hybrid import rerank_search, df
import pandas as pd

mcp = FastMCP("catalog-search")
by_id = df.set_index(df["product_id"].astype(str))

@mcp.tool()
def search_products(query: str, k: int = 5) -> list[dict]:
    """Search the furniture and home catalog. Use for any request to find products by description, style, color, material, or name."""
    results = []
    for pid in rerank_search(query, k=k):
        r = by_id.loc[pid]
        results.append({
            "product_id": pid,
            "name": r["product_name"],
            "category": r["category hierarchy"],
            "rating": None if pd.isna(r["average_rating"]) else float(r["average_rating"]),
        })
    return results

@mcp.tool()
def get_product(product_id: str) -> dict:
    """Get full details for one product by its product_id."""
    r = by_id.loc[product_id]
    return {"product_id": product_id, "name": r["product_name"],
            "category": r["category hierarchy"], "features": r["features_clean"],
            "description": r["product_description"] if r["has_description"] else None}

if __name__ == "__main__":
    mcp.run()