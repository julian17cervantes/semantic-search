import pandas as pd

MIN_CHARS = 200
MAX_CHARS = 1000
MAX_CHUNKS_PER_ARTICLE = 5
MIN_CHUNK_CHARS = 50


def chunk_text(text, max_chars=MAX_CHARS):
    chunks = []
    current = ""
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > max_chars:
            cut = paragraph.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            piece, paragraph = paragraph[:cut], paragraph[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece)
        if not paragraph:
            continue
        if current and len(current) + 1 + len(paragraph) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def build_chunks(df):
    rows = []
    for article in df.itertuples():
        text = article.text.strip()
        if len(text) < MIN_CHARS:
            continue
        for i, chunk in enumerate(chunk_text(text)[:MAX_CHUNKS_PER_ARTICLE]):
            if len(chunk) < MIN_CHUNK_CHARS:
                continue
            rows.append(
                {
                    "article_id": article.id,
                    "title": article.title,
                    "url": article.url,
                    "chunk_index": i,
                    "text": chunk,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_parquet("data/20231101.simple/train-00000-of-00001.parquet")
    chunks = build_chunks(df)
    chunks.to_parquet("data/chunks.parquet", index=False)

    print("articles in:", len(df))
    print("articles kept:", chunks["article_id"].nunique())
    print("chunks out:", len(chunks))
    print(chunks["text"].str.len().describe().round(0))