import sys

import ollama

SYSTEM_A = (
    "Answer the question using only the numbered sources. Reply in one or two sentences "
    "and cite the sources you used like [1]. If the sources do not contain the answer, "
    "reply exactly: I can't tell from the sources."
)
SYSTEM_B = (
    "Use only the numbered sources. First write a line starting with ANSWERABLE: followed by "
    "yes or no, saying whether the sources contain the answer. Then write a line starting with "
    "ANSWER: giving one or two sentences that cite the sources you used like [1]. "
    "If ANSWERABLE is no, write ANSWER: none."
)
SOURCES = (
    "[1] Canberra: Canberra is the capital city of Australia. It is in the Australian Capital Territory.\n"
    "[2] Sydney: Sydney is the largest city in Australia. It is the capital of New South Wales."
)
CASES = [
    ("What is the capital of Australia?", "should answer Canberra [1]"),
    ("Who is the prime minister of Australia?", "should decline"),
    ("Which city is the largest in Australia?", "should answer Sydney [2]"),
    ("Is Sydney the capital of Australia?", "should say no, Canberra is [1]"),
]


def ask(model, system, question):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Sources:\n{SOURCES}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0, "seed": 0, "num_predict": 120},
    )
    seconds = (getattr(response, "total_duration", None) or 0) / 1e9
    return response["message"]["content"].strip(), seconds


models = sys.argv[1:] or ["llama3.2:3b"]
for model in models:
    print("=" * 70)
    print("MODEL:", model)
    ask(model, SYSTEM_A, "warm up")
    for label, system in (("A", SYSTEM_A), ("B", SYSTEM_B)):
        print(f"\n--- prompt {label} ---")
        for question, expectation in CASES:
            answer, seconds = ask(model, system, question)
            print(f"Q: {question}   [{expectation}]")
            print(f"A: {answer}")
            print(f"   ({seconds:.1f} s)\n")