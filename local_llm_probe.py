import ollama

MODEL = "llama3.2:3b"
SYSTEM = (
    "Answer the question using only the numbered sources. Reply in one or two sentences "
    "and cite the sources you used like [1]. If the sources do not contain the answer, "
    "reply exactly: I can't tell from the sources."
)
SOURCES = (
    "[1] Canberra: Canberra is the capital city of Australia. It is in the Australian Capital Territory.\n"
    "[2] Sydney: Sydney is the largest city in Australia. It is the capital of New South Wales."
)


def ask(question):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Sources:\n{SOURCES}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0, "seed": 0, "num_predict": 120},
    )
    seconds = (getattr(response, "total_duration", None) or 0) / 1e9
    tokens = getattr(response, "eval_count", None)
    print(f"Q: {question}")
    print(f"A: {response['message']['content'].strip()}")
    print(f"   ({seconds:.1f} s, {tokens} generated tokens)\n")


ask("What is the capital of Australia?")
ask("Who is the prime minister of Australia?")