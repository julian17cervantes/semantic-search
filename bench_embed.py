import time

import pandas as pd
from sentence_transformers import SentenceTransformer

SAMPLE_SIZE = 2000

chunks = pd.read_parquet("data/chunks.parquet")
texts = chunks["text"].sample(SAMPLE_SIZE, random_state=1).tolist()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("device:", model.device)
print("max tokens per chunk:", model.max_seq_length)

start = time.time()
vectors = model.encode(texts, batch_size=64, show_progress_bar=True)
seconds = time.time() - start

rate = SAMPLE_SIZE / seconds
print("vector shape:", vectors.shape)
print(f"{rate:.0f} chunks per second")
print(f"estimated time for all {len(chunks)} chunks: {len(chunks) / rate / 60:.0f} minutes")
