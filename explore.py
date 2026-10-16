import pandas as pd

df = pd.read_parquet("data/20231101.simple/train-00000-of-00001.parquet")

print("rows:", len(df))
print("columns:", list(df.columns))

lengths = df["text"].str.len()
print("\ntext length in characters:")
print(lengths.describe().round(0))
print("articles under 200 characters:", int((lengths < 200).sum()))

sample = df.sample(1, random_state=1).iloc[0]
print("\nSAMPLE TITLE:", sample["title"])
print(sample["text"][:400])