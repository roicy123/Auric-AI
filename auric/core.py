"""Symbolic core: header parsing, constraint DSL, exact solving, conflict cores.

Nothing in here calls a model.  The model's only job (see extract.py) is to turn
each note line into one DSL constraint; everything downstream of that is exact.

Constraint DSL (all names/blocks/stations are echoed verbatim from the notes):

    AT       P B        P works block B
    NOT_AT   P B        P does not work block B
    ON       P S        P holds station S
    NOT_ON   P S        P does not hold station S
    BEFORE   P Q        P works strictly earlier than Q
    NEXT     P Q        Q works in the block immediately after P's
    BETWEEN  P Q R      P is strictly between Q and R (either order)
    S_BEFORE S P        whoever holds station S works earlier than P
    S_AFTER  S P        whoever holds station S works later than P
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

OPS = {
    "AT": ("P", "B"), "NOT_AT": ("P", "B"),
    "ON": ("P", "S"), "NOT_ON": ("P", "S"),
    "BEFORE": ("P", "P"), "NEXT": ("P", "P"),
    "BETWEEN": ("P", "P", "P"),
    "S_BEFORE": ("S", "P"), "S_AFTER": ("S", "P"),
}


@dataclass(frozen=True)
class Header:
    people: Tuple[str, ...]
    blocks: Tuple[str, ...]
    stations: Tuple[str, ...]
    holders: Tuple[str, ...]


def split_item(text: str) -> Tuple[str, List[str]]:
    """Return (header, note_lines).  The header is everything before the first
    blank line; note lines are the non-empty lines after it."""
    head, _, body = text.partition("\n\n")
    lines = [l.strip() for l in body.split("\n") if l.strip()]
    return head, lines


def parse_header(head: str) -> Header:
    """Deterministic header parse.  The header is the one part of an item that
    is structural rather than prose, so it is parsed with regexes, not a model."""
    m = re.search(r"rota:\s*(.*?)\.\s*Blocks run", head, re.S)
    people = tuple(x.strip() for x in m.group(1).split(","))
    m = re.search(r"Blocks run\s*(.*?),\s*one person per block", head, re.S)
    blocks = tuple(x.strip() for x in m.group(1).split(","))
    m = re.search(r"one person on each:\s*(.*?)\.", head, re.S)
    stations = tuple(x.strip() for x in m.group(1).split(","))
    m = re.search(r"people on a station are\s*(.*?);", head, re.S)
    holders = tuple(x.strip() for x in m.group(1).split(","))
    return Header(people, blocks, stations, holders)


# ---------------------------------------------------------------- solving
class World:
    """All candidate rotas for a header, with per-constraint satisfaction held
    as Python-int bitsets over the candidates (fast subset-SAT checks)."""

    def __init__(self, h: Header):
        self.h = h
        n = len(h.people)
        self.cands: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        pidx = {p: i for i, p in enumerate(h.people)}
        self.pidx = pidx
        for perm in itertools.permutations(range(n)):          # perm[i] = block of person i
            for sp in itertools.permutations(range(len(h.stations)), len(h.holders)):
                self.cands.append((perm, sp))                  # sp[j] = station of holder j
        self.hidx = {p: j for j, p in enumerate(h.holders)}
        self.sidx = {s: j for j, s in enumerate(h.stations)}
        self.bidx = {b: j for j, b in enumerate(h.blocks)}
        self.full = (1 << len(self.cands)) - 1
        self._cache: Dict[tuple, int] = {}

    def valid(self, op: str, args: Sequence[str]) -> bool:
        h = self.h
        kinds = OPS.get(op)
        if kinds is None or len(args) != len(kinds):
            return False
        for k, a in zip(kinds, args):
            if k == "P" and a not in self.pidx:
                return False
            if k == "B" and a not in self.bidx:
                return False
            if k == "S" and a not in self.sidx:
                return False
        if op in ("BEFORE", "NEXT") and args[0] == args[1]:
            return False
        if op == "BETWEEN" and len(set(args)) < 3:
            return False
        return True

    def mask(self, op: str, args: Tuple[str, ...]) -> int:
        key = (op, args)
        if key in self._cache:
            return self._cache[key]
        m = 0
        for ci, (perm, sp) in enumerate(self.cands):
            if self._holds(op, args, perm, sp):
                m |= 1 << ci
        self._cache[key] = m
        return m

    def _holds(self, op, a, perm, sp) -> bool:
        pi, bi, si, hi = self.pidx, self.bidx, self.sidx, self.hidx
        if op == "AT":
            return perm[pi[a[0]]] == bi[a[1]]
        if op == "NOT_AT":
            return perm[pi[a[0]]] != bi[a[1]]
        if op in ("ON", "NOT_ON"):
            on = a[0] in hi and sp[hi[a[0]]] == si[a[1]]
            return on if op == "ON" else not on
        if op == "BEFORE":
            return perm[pi[a[0]]] < perm[pi[a[1]]]
        if op == "NEXT":
            return perm[pi[a[1]]] == perm[pi[a[0]]] + 1
        if op == "BETWEEN":
            x, y, z = perm[pi[a[0]]], perm[pi[a[1]]], perm[pi[a[2]]]
            return y < x < z or z < x < y
        if op in ("S_BEFORE", "S_AFTER"):
            holder = next((p for p, j in hi.items() if sp[j] == si[a[0]]), None)
            if holder is None:
                return False
            hb, pb = perm[pi[holder]], perm[pi[a[1]]]
            return hb < pb if op == "S_BEFORE" else hb > pb
        raise ValueError(op)

    def decode(self, ci: int) -> Dict[str, Dict[str, str]]:
        perm, sp = self.cands[ci]
        out = {}
        for p, i in self.pidx.items():
            out[p] = {"block": self.h.blocks[perm[i]]}
        for p, j in self.hidx.items():
            out[p]["station"] = self.h.stations[sp[j]]
        return out


def bits(m: int) -> List[int]:
    out = []
    while m:
        low = m & -m
        out.append(low.bit_length() - 1)
        m ^= low
    return out


def popcount(m: int) -> int:
    return bin(m).count("1")


Constraint = Tuple[str, Tuple[str, ...]]          # (op, args)


def dedupe(cons: Sequence[Tuple[int, Constraint]]) -> List[Tuple[int, Constraint]]:
    """Collapse restatements: one representative line per logical constraint."""
    seen, out = set(), []
    for line, c in cons:
        if c not in seen:
            seen.add(c)
            out.append((line, c))
    return out


def classify(w: World, cons: Sequence[Constraint], max_listed: int = 4):
    """-> ('unique'|'ambiguous'|'inconsistent', [candidate indices])."""
    m = w.full
    for op, a in cons:
        m &= w.mask(op, a)
        if not m:
            break
    sols = bits(m)
    if not sols:
        return "inconsistent", []
    return ("unique" if len(sols) == 1 else "ambiguous"), sols


def minimal_cores(w: World, cons: Sequence[Tuple[int, Constraint]],
                  max_size: int = 6, limit: int = 20) -> List[List[int]]:
    """All minimal unsatisfiable subsets (as lists of line numbers), smallest
    first, found by size-ordered search with superset pruning.  `cons` must
    already be de-duplicated.  The universe is a handful of constraints, so
    exhaustive search is cheap."""
    masks = [w.mask(op, a) for _, (op, a) in cons]
    cores: List[frozenset] = []
    n = len(cons)
    for size in range(1, min(max_size, n) + 1):
        for combo in itertools.combinations(range(n), size):
            fs = frozenset(combo)
            if any(c <= fs for c in cores):
                continue
            m = w.full
            for i in combo:
                m &= masks[i]
                if not m:
                    break
            if not m:
                cores.append(fs)
                if len(cores) >= limit:
                    break
        if len(cores) >= limit:
            break
    return [[cons[i][0] for i in sorted(c)] for c in cores]
