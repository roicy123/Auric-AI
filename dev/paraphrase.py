"""DEV ONLY: build a phrasing-shifted copy of the visible set (a held-out proxy).

Every constraint line is re-worded from a template pool DISJOINT from the visible
one (new hedges, new sentence forms, new restatement openers); every distractor is
replaced by a new distractor (incl. new historical / hypothetical / wish forms).
Gold constraints are unchanged, so the key is recomputed with the exact solver.
Writes items_shift.json, key_shift.json, gold_shift.json.  Seeded -> reproducible.
"""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from auric.core import *
from dev.oracle import label

HEDGE = ["Pretty sure ", "If I'm reading the sheet right, ", "Last I checked, ", "To the best of my knowledge, ",
         "Unless the sheet is stale, ", "I seem to remember that "]
RESTATE = ["Flagging again: ", "Second mention in the log: ", "Repeating for the record: "]
T = {
    "AT": ["{P} has been given the {B} slot.", "{P} is down for {B}.", "{P}'s shift starts at {B}.", "{P} will be in at {B}."],
    "NOT_AT": ["{P} cannot work {B}.", "Nothing for {P} at {B} - off the board.", "{P} is off the board for the {B} slot."],
    "ON": ["{P} is running {S}.", "{P} staffs {S}.", "{S} is {P}'s to run.", "{P} is the {S} person."],
    "NOT_ON": ["{P} is not covering {S}.", "Keep {P} off {S}.", "{S} is not being run by {P}."],
    "BEFORE": ["{P} starts earlier than {Q} does.", "{P} goes in ahead of {Q}.", "{Q} starts after {P} has gone in."],
    "NEXT": ["{Q} follows {P} with nothing in between.", "{Q} comes in straight after {P}.", "{P}, and right behind them {Q}."],
    "BETWEEN": ["{P} falls in the gap between {Q} and {R}, whichever order they go.", "Of {P}, {Q} and {R}, {P} is the middle one.",
                "{Q} and {R} bracket {P}, one on each side."],
    "S_BEFORE": ["Whoever is running {S} goes in before {P}.", "{P} comes after the {S} person.", "{S} is staffed ahead of {P}'s start."],
}
DIST_NOP = ["The kettle in the break room needs descaling.", "Forklift charging bays are being repainted.", "The fire exit signage was replaced on Tuesday.",
            "Parking validation now runs through the front office.", "Someone left a umbrella by the loading door.",
            "The shift handover template got a new header last week."]
DIST_P = ["{P} was moved off {S} last year.", "{P} would have liked the {B} slot, but that fell through.", "Whether {P} takes {B} is still undecided.",
          "If {P} were on {S}, the audit would be easier.", "{P} borrowed a torque wrench from the tool crib.", "{P} used to open up at {B} before the reorganisation.",
          "{P} asked about {S} but was told to wait for next quarter.", "{P} and {Q} swapped lockers.", "It was suggested that {P} might suit {S}; nobody followed up."]


def lc(s, h):
    first = s.split()[0].strip("'s,").replace("'s", "")
    return s if any(s.startswith(p) for p in h.people) else s[0].lower() + s[1:]


def render(c, rng, h):
    op, a = c
    if op in ("AT", "NOT_AT"): d = dict(P=a[0], B=a[1])
    elif op in ("ON", "NOT_ON"): d = dict(P=a[0], S=a[1])
    elif op in ("BEFORE", "NEXT"): d = dict(P=a[0], Q=a[1])
    elif op == "BETWEEN": d = dict(P=a[0], Q=a[1], R=a[2])
    else: d = dict(S=a[0], P=a[1])
    return rng.choice(T[op]).format(**d)


def main():
    rng = random.Random(7)
    items = json.loads((ROOT / "items.json").read_text())
    out, key, gold = [], {}, {}
    for it in items:
        head, lines = split_item(it["text"]); h = parse_header(head); w = World(h)
        new, cons, g = [], [], []
        for i, l in enumerate(lines):
            c = label(l, h)
            if c:
                s = render(c, rng, h)
                if rng.random() < 0.55: s = rng.choice(HEDGE) + lc(s, h)
                if rng.random() < 0.10: s = rng.choice(RESTATE) + s
                new.append(s); cons.append((i, c)); g.append(c)
            else:
                t = rng.choice(DIST_P + DIST_NOP)
                p = rng.sample(h.people, 2)
                s = t.format(P=p[0], Q=p[1], S=rng.choice(h.stations), B=rng.choice(h.blocks))
                if rng.random() < 0.35: s = rng.choice(HEDGE) + lc(s, h)
                new.append(s); g.append(None)
        case, sols = classify(w, [c for _, c in dedupe(cons)])
        k = {"case": case, "assignments": [w.decode(s) for s in sols] if case != "inconsistent" else [], "conflict_line_sets": []}
        if case == "inconsistent":
            k["conflict_line_sets"] = [[new[i] for i in c] for c in minimal_cores(w, cons, limit=200)]
        key[it["id"]] = k
        gold[it["id"]] = [[c[0], list(c[1])] if c else None for c in g]
        out.append({**it, "text": head + "\n\n" + "\n".join(new)})
    (ROOT / "items_shift.json").write_text(json.dumps(out, indent=1))
    (ROOT / "key_shift.json").write_text(json.dumps(key, indent=1))
    (ROOT / "gold_shift.json").write_text(json.dumps(gold))
    print("wrote shifted set:", len(out))


if __name__ == "__main__":
    main()
