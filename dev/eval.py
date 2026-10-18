"""DEV: run the pipeline on items.json, score with score.py, report kind matrix + line-level accuracy.
  python3 dev/eval.py --budget 1x [--fake 0.1] [--runs 3] [--ablate vote,repair]
"""
import argparse, json, sys, collections, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from auric.cli import run_items
from auric.pipeline import Flags
from auric.core import split_item, parse_header
from dev.oracle import label


def load_score():
    for p in [ROOT / "score.py"]:
        spec = importlib.util.spec_from_file_location("score", p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="1x"); ap.add_argument("--items", default=str(ROOT / "items.json"))
    ap.add_argument("--key", default=str(ROOT / "visible_key.json"))
    ap.add_argument("--fake", type=float, default=None); ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--ablate", default=""); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save", default=None); ap.add_argument("--gold", default=None)
    a = ap.parse_args()
    items = json.load(open(a.items)); key = json.load(open(a.key)); S = load_score()
    gold = json.load(open(a.gold)) if a.gold else None
    flags = Flags(**{k: False for k in a.ablate.split(",") if k})
    for r in range(a.runs):
        if a.fake is not None:
            from dev.fake import FakeClient
            client = FakeClient(a.fake, seed=r)
        else:
            from auric.extract import ModelClient
            client = ModelClient(ROOT / ".env")
        traces = {}
        res = run_items(items, client, a.budget, flags, a.workers, traces)
        if a.save: Path(f"{a.save}.run{r}.json").write_text(json.dumps(res, indent=1))
        # budget audit (per item, as the grader's proxy would see it)
        cap = {"1x": 1, "3x": 3, "10x": 10}[a.budget]
        calls = {i: t.get("calls", 0) for i, t in traces.items()}
        viol = [i for i, c in calls.items() if c < 1 or c > cap]
        # line-level accuracy vs oracle labels (first sample only = what 1x sees)
        tot = ok = 0
        for it in items:
            head, lines = split_item(it["text"]); h = parse_header(head)
            s0 = (traces[it["id"]].get("samples") or [{}])[0]
            for i, l in enumerate(lines, 1):
                g = label(l, h) if gold is None else ((lambda x: (x[0], tuple(x[1])) if x else None)(gold[it['id']][i-1]))
                if g is not None and g[0] == "BETWEEN": g = (g[0], (g[1][0],) + tuple(sorted(g[1][1:])))
                tot += 1; ok += (s0.get(i) == g)
        # ---- diagnostics: per gold-op accuracy + sample mistakes (first sample)
        st = collections.defaultdict(lambda: [0, 0]); wrong = collections.defaultdict(list)
        for it in items:
            head, lines = split_item(it["text"]); h = parse_header(head)
            s0 = (traces[it["id"]].get("samples") or [{}])[0]
            for i, l in enumerate(lines, 1):
                g = label(l, h) if gold is None else ((lambda x: (x[0], tuple(x[1])) if x else None)(gold[it['id']][i-1]))
                if g is not None and g[0] == "BETWEEN": g = (g[0], (g[1][0],) + tuple(sorted(g[1][1:])))
                k = g[0] if g else "NONE"; st[k][0] += 1
                if s0.get(i) == g: st[k][1] += 1
                else: wrong[k].append((it["id"], l, g, s0.get(i)))
        rep = ["per gold op: n, accuracy"] + [f"  {k:9s} n={v[0]:4d} acc={v[1]/v[0]:.2f}" for k, v in sorted(st.items())]
        for k, v in sorted(wrong.items()):
            rep.append(f"-- mistakes where gold={k} ({len(v)}), first 8:")
            for x in v[:8]: rep.append(f"   {x[0]} | {x[1]}\n        gold={x[2]}  got={x[3]}")
        Path(f"diag_{a.budget}_run{r}.txt").write_text("\n".join(rep), encoding="utf-8")
        print("\n".join(rep[:rep.index([z for z in rep if z.startswith('-- ')][0]) if any(z.startswith('-- ') for z in rep) else len(rep)]))
        if hasattr(client, "raw"): Path(f"raw_{a.budget}_run{r}.json").write_text(json.dumps(client.raw, indent=1), encoding="utf-8")
        kinds = collections.Counter((key[i]["case"], res[i]["case"]) for i in res)
        out = S.score(key, res, items) if hasattr(S, "score") else None
        print(f"run {r}: budget {a.budget} flags-off={a.ablate or '-'} budget-violations={len(viol)} "
              f"mean-calls={sum(calls.values())/len(calls):.2f} line-acc(first sample)={ok/tot:.3f}")
        print("  kind matrix (true,declared):", dict(kinds))
        if isinstance(out, dict):
            print("  score.py:", {k: v for k, v in out.items() if not isinstance(v, (dict, list))})

if __name__ == "__main__":
    main()
