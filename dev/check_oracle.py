"""DEV ONLY: do oracle labels + exact solver reproduce visible_key.json?"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from auric.core import *
from dev.oracle import label

root = Path(__file__).resolve().parent.parent
items = json.loads((root / "items.json").read_text())
key = json.loads((root / "visible_key.json").read_text())
worlds = {}
bad = 0
conf = collections.Counter()
unlabeled = collections.Counter()
for it in items:
    head, lines = split_item(it["text"])
    h = parse_header(head)
    w = worlds.setdefault(h, World(h))
    cons = []
    for i, l in enumerate(lines):
        c = label(l, h)
        if c is None:
            unlabeled[l] += 1
        elif w.valid(*c):
            cons.append((i, c))
        else:
            print("INVALID", it["id"], l, c)
    case, sols = classify(w, [c for _, c in dedupe(cons)])
    k = key[it["id"]]
    ok = case == k["case"]
    if ok and case in ("unique", "ambiguous"):
        got = sorted(json.dumps(w.decode(s), sort_keys=True) for s in sols)
        want = sorted(json.dumps(a, sort_keys=True) for a in k["assignments"])
        # key assignments may omit station for non-holders; compare decode (also omits)
        ok = got == want
    if ok and case == "inconsistent":
        cores = minimal_cores(w, dedupe(cons))
        want = k["conflict_line_sets"]
        ok = len(cores) > 0
    conf[(k["case"], case)] += 1
    if not ok:
        bad += 1
        print("MISMATCH", it["id"], "key:", k["case"], "got:", case, len(sols))
print("mismatches", bad, "/", len(items))
print(conf)
print("unlabeled distinct lines:", len(unlabeled))
