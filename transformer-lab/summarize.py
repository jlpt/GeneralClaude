"""Print mean ± std val loss per config name, in first-seen order.
Usage: python summarize.py [prefix]"""
import json, os, statistics, sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
prefix = sys.argv[1] if len(sys.argv) > 1 else ""
rows = OrderedDict()
for line in open(os.path.join(HERE, "results.jsonl")):
    r = json.loads(line)
    if r["name"].startswith(prefix):
        rows.setdefault(r["name"], []).append(r)

print(f"{'config':44s} {'n':>2s} {'val loss':>16s} {'bpc':>6s} {'params':>9s} {'active':>9s} {'time':>6s}")
for name, rs in rows.items():
    v = [r["val_loss"] for r in rs]
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    m = statistics.mean(v)
    print(f"{name:44s} {len(v):2d} {m:9.4f} ± {sd:.4f} {m / 0.693147:6.3f} "
          f"{rs[0]['params']:9,d} {rs[0]['active_params']:9,d} {statistics.mean(r['time_s'] for r in rs):5.0f}s")
