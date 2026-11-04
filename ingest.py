import pandas as pd

df = pd.read_csv("data/WANDS/dataset/product.csv", sep="\t")
# print(df.columns.tolist())   # all column names
# print(df.isna().sum())       # missing count for every column
# print(df.describe(include="all"))  # summary including text columns

# print("Missing before:\n", df.isna().sum())

# print((df["category hierarchy"].str.strip() == "").sum())

df["has_description"] = df["product_description"].notna()
df["category hierarchy"] = (
    df["category hierarchy"].fillna(df["product_class"]).fillna("Uncategorized")
)

def clean_features(s):
    if pd.isna(s):
        return ""
    out, seen = [], set()
    for p in str(s).split("|"):
        if ":" not in p:
            continue
        k, v = (x.strip() for x in p.split(":", 1))
        if not k or not v or v.lower() == "no":
            continue
        if (k, v) in seen:
            continue
        seen.add((k, v))
        out.append(f"{k}: {v}")
        # print(out)
    return "; ".join(out)

df["features_clean"] = df["product_features"].apply(clean_features)

df["search_text"] = (
    df["product_name"].fillna("") + ". "
    + df["category hierarchy"] + ". "
    + df["features_clean"] + ". "
    + df["product_description"].fillna("")
)

df = df.drop_duplicates(subset=["product_id"])

print("Missing after:\n", df.isna().sum())
print("Products:", len(df))
print("\nSample:\n", df.loc[0, "search_text"][:500])

df.to_parquet("data/products_clean.parquet", index=False)
