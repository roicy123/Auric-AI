"""./run <items.json> --budget 1x|3x|10x --out answers.json"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .extract import ModelClient
from .pipeline import Flags, solve_item

ROOT = Path(__file__).resolve().parent.parent


def run_items(items, client, budget, flags=None, workers=6, traces=None):
    def work(it):
        tr = {} if traces is not None else None
        try:
            a = solve_item(it, client, budget, flags, tr)
        except Exception as exc:                      # never lose the whole run to one item
            print(f"{it['id']}: ERROR {exc!r}", file=sys.stderr)
            a = {"case": "inconsistent", "conflicts": []}
        if traces is not None:
            traces[it["id"]] = tr
        return it["id"], a
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(work, items))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("items")
    ap.add_argument("--budget", choices=["1x", "3x", "10x"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="dev only")
    a = ap.parse_args(argv)
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))
    if a.limit:
        items = items[:a.limit]
    client = ModelClient(ROOT / ".env")
    res = run_items(items, client, a.budget, workers=a.workers)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(res)} answers; {client.total_calls} model calls", file=sys.stderr)
