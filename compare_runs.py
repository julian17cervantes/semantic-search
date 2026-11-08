import json
import sys

from retrieval_lib import hit_at, mrr, paired_bootstrap_diff

with open(sys.argv[1]) as f:
    base = json.load(f)
with open(sys.argv[2]) as f:
    new = json.load(f)
assert base["queries"] == new["queries"], "the two runs used different queries"

metrics = {"hit@10": lambda r: hit_at(r, 10), "MRR": mrr}
print(f"{'method':>9} {'metric':>7} {'baseline':>9} {'new':>7} {'change':>8}  95% interval of the change")
for method in base["ranks"]:
    for name, fn in metrics.items():
        a, b = base["ranks"][method], new["ranks"][method]
        diff, low, high = paired_bootstrap_diff(a, b, fn)
        print(f"{method:>9} {name:>7} {fn(a):>9.3f} {fn(b):>7.3f} {diff:>+8.3f}  [{low:+.3f}, {high:+.3f}]")


def key(rank):
    return 99 if rank is None else rank


a, b = base["ranks"]["semantic"], new["ranks"]["semantic"]
better = [q for q, x, y in zip(base["queries"], a, b) if key(y) < key(x)]
worse = [q for q, x, y in zip(base["queries"], a, b) if key(y) > key(x)]
print(f"\nsemantic: {len(better)} queries ranked higher, {len(worse)} lower, {len(a) - len(better) - len(worse)} unchanged")
for label, group in (("HIGHER", better), ("LOWER", worse)):
    for query in group:
        print(f"  {label}: {query}")