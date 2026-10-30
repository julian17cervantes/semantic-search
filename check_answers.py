import pandas as pd
import sys

chunks = pd.read_parquet("data/chunks.parquet", columns=["text"])
path = sys.argv[1] if len(sys.argv) > 1 else "eval/answers_dev.csv"
rows = pd.read_csv(path)

print(f"{'chunks':>7}  question -> accepted answers")
for question, answer in zip(rows["question"], rows["answer"]):
    golds = answer.split("|")
    counts = [int(chunks["text"].str.contains(g, case=False, regex=False).sum()) for g in golds]
    flag = "" if max(counts) > 0 else "   <-- NOT FOUND IN CORPUS"
    print(f"{max(counts):>7}  {question} -> {answer}{flag}")