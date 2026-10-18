import pandas as pd

chunks = pd.read_parquet("data/chunks.parquet", columns=["title"])
titles = set(chunks["title"])
queries = pd.read_csv("eval/queries.csv")

print("queries:", len(queries))
for _, row in queries.iterrows():
    found = [t for t in row["expected"].split("|") if t in titles]
    if not found:
        print("NO MATCH:", row["query"], "->", row["expected"])
        close = sorted(t for t in titles if row["expected"].split("|")[0].lower() in t.lower())
        print("   similar titles:", close[:5])