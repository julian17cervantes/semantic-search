import pandas as pd

chunks = pd.read_parquet("data/chunks.parquet", columns=["text"])
rows = pd.read_csv("eval/answers_dev.csv")

print(f"{'chunks':>7}  question -> accepted answers")
for question, answer in zip(rows["question"], rows["answer"]):
    golds = answer.split("|")
    counts = [int(chunks["text"].str.contains(g, case=False, regex=False).sum()) for g in golds]
    flag = "" if max(counts) > 0 else "   <-- NOT FOUND IN CORPUS"
    print(f"{max(counts):>7}  {question} -> {answer}{flag}")