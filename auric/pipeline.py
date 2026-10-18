"""Per-item pipeline: extract (model) -> vote -> solve (exact) -> answer.

Components can be switched off for the ablation table via `Flags`.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .core import (World, classify, dedupe, minimal_cores, parse_header, split_item)
from .extract import ItemBudget, ask

BUDGETS = {"1x": 1, "3x": 3, "10x": 10}
_WORLDS: Dict[object, World] = {}


@dataclass
class Flags:
    vote: bool = True          # per-line majority vote across samples (needs budget > 1)
    adaptive: bool = True      # stop sampling early once two samples agree and result is healthy
    repair: bool = True        # use leftover budget on single-call re-asks of disputed lines
    health: bool = True        # use solver-side plausibility to choose among candidate answers
    cite_by_conf: bool = True  # among minimal cores pick the one the model was most consistent on


def world_for(h) -> World:
    if h not in _WORLDS:
        _WORLDS[h] = World(h)
    return _WORLDS[h]


def healthy(case: str, sols: Sequence[int], cores: Optional[List[List[int]]]) -> bool:
    """Priors from the brief: ambiguous has 2-4 solutions; every minimal conflict has >= 3 lines."""
    if case == "unique":
        return True
    if case == "ambiguous":
        return 2 <= len(sols) <= 4
    return bool(cores) and min(len(c) for c in cores) >= 3


def evaluate(w: World, labels: Dict[int, Optional[tuple]]):
    cons = dedupe([(i, c) for i, c in sorted(labels.items()) if c is not None])
    case, sols = classify(w, [c for _, c in cons])
    cores = minimal_cores(w, cons) if case == "inconsistent" else None
    return case, sols, cores, cons


def vote(samples: List[Dict[int, Optional[tuple]]], nums: Sequence[int]):
    """Per line: most common label (None counts as a label).  Ties favour NONE-free
    labels? No -- ties favour the earliest sample, keeping the rule simple.
    Returns (labels, agreement) with agreement[n] = share of samples for the winner."""
    labels, agree = {}, {}
    for n in nums:
        votes = collections.Counter(s.get(n) for s in samples)
        best = max(votes.items(), key=lambda kv: (kv[1], -[s.get(n) for s in samples].index(kv[0])))
        labels[n], agree[n] = best[0], best[1] / len(samples)
    return labels, agree


def build_answer(it_lines: Sequence[str], w: World, labels, agree, flags: Flags):
    case, sols, cores, cons = evaluate(w, labels)
    if case == "unique":
        return {"case": "unique", "assignment": w.decode(sols[0])}
    if case == "ambiguous":
        return {"case": "ambiguous", "assignments": [w.decode(s) for s in sols[:4]]}
    if cores:
        def conf(c):
            return sum(agree.get(i + 1, 1.0) for i in c) / len(c)
        pool = [c for c in cores if len(c) >= 3] or cores
        pick = max(pool, key=conf) if flags.cite_by_conf else pool[0]
        return {"case": "inconsistent", "conflicts": [it_lines[i - 1] for i in sorted(pick)]}
    return {"case": "inconsistent", "conflicts": []}


def solve_item(item: dict, client, budget_name: str, flags: Optional[Flags] = None, trace: Optional[dict] = None):
    flags = flags or Flags()
    head, lines = split_item(item["text"])
    h = parse_header(head)
    w = world_for(h)
    nums = list(range(1, len(lines) + 1))
    budget = ItemBudget(BUDGETS[budget_name])
    samples: List[Dict[int, Optional[tuple]]] = []

    def one_sample():
        r = ask(client, budget, item["id"], h, w, lines, nums)
        if r is not None:
            samples.append(r)
        return r

    one_sample()                                   # every item gets >= 1 model call
    while budget.left() > 0 and flags.vote:
        if flags.adaptive and len(samples) >= 2 and samples[-1] == samples[-2]:
            cur = evaluate(w, samples[-1])
            if not flags.health or healthy(*cur[:3]):
                break
        if flags.adaptive and len(samples) >= 3:
            labels, _ = vote(samples, nums)
            cur = evaluate(w, labels)
            if healthy(*cur[:3]) and all(
                    sum(1 for s in samples if s.get(n) == labels[n]) / len(samples) >= 2 / 3 for n in nums):
                break
        if not one_sample() and budget.left() == 0:
            break

    if not samples:                                # every call failed: fall back to an empty extraction
        samples = [{n: None for n in nums}]
    labels, agree = vote(samples, nums) if flags.vote else (samples[0], {n: 1.0 for n in nums})

    # Repair: re-ask only the disputed lines (short prompt, one call), fold result in as extra votes.
    if flags.repair and flags.vote and budget.left() > 0:
        disputed = [n for n in nums if agree[n] < 1.0]
        if disputed:
            r = ask(client, budget, item["id"], h, w, [lines[n - 1] for n in disputed], disputed)
            if r is not None:
                # the focused answer counts as two votes for each disputed line
                votes = {n: collections.Counter(s.get(n) for s in samples) for n in disputed}
                for n in disputed:
                    votes[n][r.get(n)] += 2
                    best = votes[n].most_common(1)[0]
                    labels[n], agree[n] = best[0], best[1] / (len(samples) + 2)

    ans = build_answer(lines, w, labels, agree, flags)

    # Health-based fallback: if the voted result is implausible, prefer the most common plausible per-sample result.
    if flags.health and len(samples) > 1:
        case, sols, cores, _ = evaluate(w, labels)
        if not healthy(case, sols, cores):
            tally = collections.Counter()
            best_ans = {}
            for s in samples:
                c, so, co, _ = evaluate(w, s)
                if healthy(c, so, co):
                    a = build_answer(lines, w, s, {n: 1.0 for n in nums}, flags)
                    k = repr(sorted(a.items(), key=str))
                    tally[k] += 1
                    best_ans[k] = a
            if tally:
                ans = best_ans[tally.most_common(1)[0][0]]
    if trace is not None:
        trace.update(calls=budget.used, samples=samples, labels=labels, agree=agree)
    return ans
