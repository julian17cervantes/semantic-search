import re
import time

ABSTAIN = "I can't tell from the sources."

SYSTEM_PROMPT = (
    "Answer the question using only the numbered sources. Reply in one or two sentences "
    "and cite the sources you used like [1]. If the sources do not contain the answer, "
    "reply exactly: " + ABSTAIN
)

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _squash(text):
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(text.strip().strip("\"'`").split()).casefold()


_ABSTAIN_START = _squash(ABSTAIN).rstrip(".")


def build_messages(question, hits, system_prompt=SYSTEM_PROMPT):
    sources = "\n".join(f"[{i}] {h['title']}: {h['text']}" for i, h in enumerate(hits, 1))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
    ]


def parse_answer(raw, num_sources):
    """Decide whether the model abstained, and which of the numbered sources it cited."""
    text = raw.strip()
    if _squash(text).startswith(_ABSTAIN_START):
        # Anything after the fallback sentence is ignored, including citations
        return {"abstained": True, "answer": None, "citations": [], "invalid_citations": [], "has_citation": False}
    cited = []
    for group in _CITATION.findall(text):
        for number in group.split(","):
            n = int(number)
            if n not in cited:
                cited.append(n)
    valid = sorted(n for n in cited if 1 <= n <= num_sources)
    invalid = sorted(n for n in cited if not 1 <= n <= num_sources)
    return {
        "abstained": False,
        "answer": text,
        "citations": valid,
        "invalid_citations": invalid,
        "has_citation": bool(valid),
    }


class OllamaClient:
    """Calls a local model through Ollama. Returns (text, info)."""

    def __init__(self, model, num_predict=120, seed=0):
        self.model = model
        self.num_predict = num_predict
        self.seed = seed

    def generate(self, messages):
        import ollama

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0, "seed": self.seed, "num_predict": self.num_predict},
        )
        info = {
            "model": self.model,
            "seconds": (getattr(response, "total_duration", None) or 0) / 1e9,
            "prompt_tokens": getattr(response, "prompt_eval_count", None),
            "output_tokens": getattr(response, "eval_count", None),
        }
        return response["message"]["content"], info


class Rag:
    def __init__(self, retrieve, llm, system_prompt=SYSTEM_PROMPT):
        """retrieve(question) returns hits (dicts with title, url, text); llm.generate(messages) returns (text, info)."""
        self.retrieve = retrieve
        self.llm = llm
        self.system_prompt = system_prompt

    def answer(self, question):
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        retrieve_ms = round((time.perf_counter() - t0) * 1000, 2)
        base = {
            "question": question,
            "sources": [{"title": h["title"], "url": h["url"]} for h in hits],
            "retrieve_ms": retrieve_ms,
        }
        if not hits:
            return {**base, **parse_answer(ABSTAIN, 0), "raw": ABSTAIN, "llm_ms": 0.0, "llm": {}}

        messages = build_messages(question, hits, self.system_prompt)
        t1 = time.perf_counter()
        raw, info = self.llm.generate(messages)
        llm_ms = round((time.perf_counter() - t1) * 1000, 2)
        return {**base, **parse_answer(raw, len(hits)), "raw": raw, "llm_ms": llm_ms, "llm": info}