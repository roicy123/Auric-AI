"""Model-facing layer: prompt, call client, tolerant parser, deterministic repair.

The weak model does ONE thing: read numbered note lines and say, per line, which
formal constraint (if any) the line asserts.  It never solves, never counts
solutions, never quotes text -- all of that is symbolic (core.py / pipeline.py).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .core import Header, World, OPS

# --------------------------------------------------------------------- prompt
SYSTEM = ("You convert shift-note sentences into formal scheduling constraints. "
          "You never solve the puzzle. Output only the requested lines.")

OP_HELP = """\
Constraint types (use these exact words):
  AT        | person | block            the person works that block
  NOT_AT    | person | block            the person does NOT work that block
  ON        | person | station          the person is on that station
  NOT_ON    | person | station          the person is NOT on that station
  BEFORE    | earlier | later            first person works in an earlier block than the second
  NEXT      | first | second             second person works in the block IMMEDIATELY after the first (adjacent, no gap)
  BETWEEN   | middle | end1 | end2      middle person's block is somewhere between the two others (either order)
  S_BEFORE  | station | person          whoever is on that station works EARLIER than the person
  S_AFTER   | station | person          whoever is on that station works LATER than the person
  NONE                                  the line does not constrain the rota"""

RULES = """\
How to decide:
- A line is a constraint ONLY if it states how the rota actually is: who works which block, who is on which station, or how people's blocks are ordered.
- A hedge ("I'm fairly sure", "as far as I know", "speaking from memory", "my recollection is", "going off the roster") does NOT weaken a statement. Ignore the hedge and label the statement underneath. The same goes for "noted twice", "it bears repeating" and similar openers.
- A line that merely restates another line is labelled exactly like it.
- NONE: past or former arrangements ("last month", "back in the old arrangement", "in the spring ... held"), what someone wished for, asked for or was refused, hypotheticals ("if ...had", "would have been", "had the rota gone the other way"), things nobody could remember or that were left open, and all background (surveys, keys, parking, deliveries, car-sharing, training, notes, gossip).
- Names, blocks and stations must be copied exactly as listed in the ROTA."""

EXAMPLES = """\
Example (different people and wording from your real notes):
ROTA people: Ana, Ben, Cho, Dev, Eli | blocks: 07:00, 09:00, 11:00, 13:00, 15:00 | stations: intake, calibration, packing
NOTES:
1. Dev is rostered for 13:00.
2. Honestly I think Ben was once on packing, years back.
3. Ana can't do the 07:00 slot.
4. Cho hands over directly to Eli.
5. Whoever has packing starts before Dev.
6. If Eli had been given 09:00 things would have run better.
7. Intake belongs to Ana.
8. Ben's slot is somewhere between Cho's and Dev's.
9. Calibration is not Cho's job.
10. The vending machine is out of order again.
11. Eli is on later than whoever is on intake.
OUTPUT:
1 | AT | Dev | 13:00
2 | NONE
3 | NOT_AT | Ana | 07:00
4 | NEXT | Cho | Eli
5 | S_BEFORE | packing | Dev
6 | NONE
7 | ON | Ana | intake
8 | BETWEEN | Ben | Cho | Dev
9 | NOT_ON | Cho | calibration
10 | NONE
11 | S_BEFORE | intake | Eli"""

V2_CHEATSHEET = """\
How common wordings map to types (argument ORDER matters):
  ON:       "<station> is covered by X" / "the <station> station is down to X" / "<station> is X's station" / "X has <station> this week"  -> ON | X | <station>
  NOT_ON:   "<station> is covered by someone other than X" / "you can rule X out for <station>" / "X has not been put on <station>" -> NOT_ON | X | <station>
  AT:       "X takes 11:00" / "11:00 is when X is scheduled" / "the 11:00 block is X's" / "whoever drew the 11:00 block, it was X" / "X is the one who opens up at 11:00" -> AT | X | 11:00
  NOT_AT:   "X is unavailable at 11:00" / "X has a standing commitment that rules out 11:00" / "you will not find X on the 11:00 block" -> NOT_AT | X | 11:00
  BEFORE:   "X works earlier than Y" / "X is done before Y starts" / "Y comes later in the day than X" / "by the time Y starts, X has already been on" / "Y takes over from X later in the day" -> BEFORE | X | Y   (X is the EARLIER one; this does NOT say they are adjacent)
  NEXT:     "X then Y, back to back" / "X hands straight over to Y" / "X is on the block immediately before Y" / "Y is on the block directly after X" / "Y relieves X directly" / "no block between X's and Y's, in that order" -> NEXT | X | Y   (adjacent blocks, X first)
  BETWEEN:  "X sits between Y and Z" / "put X between Y and Z" / "X is on after one of Y and Z and before the other" / "whichever way round Y and Z are, X is between them" -> BETWEEN | X | Y | Z   (the MIDDLE person is written FIRST)
  S_BEFORE: "<station> is covered before X comes on" / "whoever has <station> is done before X starts" / "the person on <station> precedes X" / "X is on later than whoever has <station>" -> S_BEFORE | <station> | X   (the station holder is EARLIER than X)
  S_AFTER:  only when the station holder is LATER than X, e.g. "X is on earlier than whoever has <station>" -> S_AFTER | <station> | X
NOT constraints (label NONE): "X wanted 07:00 and did not get it", "X put in for 15:00 and was turned down", "X asked to move to 09:00; the request was declined", "the question of X on 11:00 was left open", "nobody could remember whether X was on 09:00", "in the spring X held intake", "last month X was on packing", "back in the old arrangement X took packing", "if X had taken 11:00 ...", "had the rota gone the other way, X would be on 13:00", "X would have been a better fit for 09:00". Someone WANTING or being REFUSED a block says nothing about where they are, so it is NONE."""

GLOSS_EXAMPLE_NOTE = ("(Before each constraint you may not add anything: one output line per note, "
                      "format exactly `number | TYPE | arguments`.)")


VARIANT = os.environ.get("AURIC_VARIANT", "v2")


def build_prompt(header: Header, lines: Sequence[str], numbers: Optional[Sequence[int]] = None) -> str:
    if VARIANT == "v2":
        return build_prompt_v2(header, lines, numbers)
    nums = list(numbers) if numbers is not None else list(range(1, len(lines) + 1))
    notes = "\n".join(f"{n}. {t}" for n, t in zip(nums, lines))
    rota = (f"ROTA people: {', '.join(header.people)} | blocks (earliest to latest): {', '.join(header.blocks)} "
            f"| stations: {', '.join(header.stations)} | people on a station: {', '.join(header.holders)}")
    return (f"{OP_HELP}\n\n{RULES}\n\n{EXAMPLES}\n\n"
            f"Now do the real task. For EACH numbered note output exactly one line: "
            f"`number | TYPE | arguments` (or `number | NONE`). Output nothing else.\n\n"
            f"{rota}\nNOTES:\n{notes}\nOUTPUT:\n")


def build_prompt_v2(header, lines, numbers=None):
    nums = list(numbers) if numbers is not None else list(range(1, len(lines) + 1))
    notes = "\n".join(f"{n}. {t}" for n, t in zip(nums, lines))
    rota = (f"ROTA people: {', '.join(header.people)} | blocks (earliest to latest): {', '.join(header.blocks)} "
            f"| stations: {', '.join(header.stations)} | people on a station: {', '.join(header.holders)}")
    ex = EXAMPLES.split("OUTPUT:")[0] + """OUTPUT:
1 | Dev works 13:00 | AT | Dev | 13:00
2 | Ben held packing years ago, not now | NONE
3 | Ana cannot work 07:00 | NOT_AT | Ana | 07:00
4 | Cho hands over directly to Eli, so Eli is right after Cho | NEXT | Cho | Eli
5 | the packing person is earlier than Dev | S_BEFORE | packing | Dev
6 | hypothetical about Eli, nothing is stated | NONE
7 | Ana is on intake | ON | Ana | intake
8 | Ben is between Cho and Dev | BETWEEN | Ben | Cho | Dev
9 | Cho is not on calibration | NOT_ON | Cho | calibration
10 | trivia | NONE
11 | Eli is later than the intake person, so the intake person is earlier than Eli | S_BEFORE | intake | Eli"""
    return (f"{OP_HELP}\n\n{RULES}\n\n{V2_CHEATSHEET}\n\n{ex}\n\n"
            f"Now do the real task. Output EXACTLY ONE line for EVERY numbered note, in order, never skipping one: "
            f"`number | what the note says in plain words | TYPE | arguments` (or `number | plain words | NONE`). "
            f"Output nothing else.\n\n{rota}\nNOTES:\n{notes}\nOUTPUT:\n")


# --------------------------------------------------------------------- parsing
_OPS = set(OPS) | {"NONE"}
_ALIASES = {"NOT_ON_STATION": "NOT_ON", "NOTON": "NOT_ON", "NOTAT": "NOT_AT", "STATION_BEFORE": "S_BEFORE",
            "STATION_AFTER": "S_AFTER", "SBEFORE": "S_BEFORE", "SAFTER": "S_AFTER", "IMMEDIATELY_AFTER": "NEXT"}


def canon(op: str, args: Tuple[str, ...]):
    if op == "BETWEEN":
        return (op, (args[0],) + tuple(sorted(args[1:])))
    return (op, tuple(args))


def _norm_token(tok: str, header: Header) -> Optional[str]:
    t = tok.strip().strip("`*'\"[]()").rstrip(".,;")
    t = re.sub(r"'s$", "", t).strip()
    for p in header.people:
        if t.casefold() == p.casefold():
            return p
    for s in header.stations:
        if t.casefold() == s.casefold():
            return s
    m = re.fullmatch(r"(\d{1,2}):(\d\d)", t)
    if m:
        b = f"{int(m.group(1)):02d}:{m.group(2)}"
        if b in header.blocks:
            return b
    return t if t in header.blocks else None


_OPRE = re.compile(r"\b(NOT_AT|NOT_ON|S_BEFORE|S_AFTER|BETWEEN|BEFORE|NEXT|AT|ON|NONE)\b")


def parse_output(text: str, header: Header, w: World, expect: Sequence[int]) -> Dict[int, Optional[tuple]]:
    """-> {line_number: canonical (op, args) | None}.  Missing/garbled/invalid -> None.
    Lenient: the operator is located anywhere after the line number (models often
    echo the sentence or a gloss first), arguments are the next valid tokens, and a
    swapped argument order is repaired when the argument *types* make the intent clear."""
    out: Dict[int, Optional[tuple]] = {n: None for n in expect}
    for raw in text.splitlines():
        m = re.match(r"^\s*[`*]*\s*(\d+)\s*[.:)|-]?\s*(.*)$", raw)
        if not m:
            continue
        n = int(m.group(1))
        if n not in out:
            continue
        rest = m.group(2)
        om = _OPRE.search(rest)
        if om is None:                                   # legacy: lowercase / alias spelling
            first = re.sub(r"[^A-Za-z_ ]", "", rest.split("|")[0]).strip().upper().replace(" ", "_")
            op = _ALIASES.get(first, first)
            if op not in OPS:
                continue
            tail = "|".join(rest.split("|")[1:])
        else:
            op, tail = om.group(1), rest[om.end():]
        if op == "NONE":
            continue
        kinds = OPS[op]
        toks = []
        for chunk in re.split(r"[|,;()\[\]\s]+", tail):
            if not chunk:
                continue
            t = _norm_token(chunk, header)
            if t is not None:
                toks.append(t)
            if len(toks) == len(kinds):
                break
        if len(toks) != len(kinds):
            continue

        def kind_of(t):
            return "P" if t in header.people else "S" if t in header.stations else "B" if t in header.blocks else "?"
        have = tuple(kind_of(t) for t in toks)
        if have != kinds and sorted(have) == sorted(kinds) and len(kinds) == 2:
            toks = toks[::-1]
        args = tuple(toks)
        if w.valid(op, args):
            out[n] = canon(op, args)
    return out


# ---------------------------------------------------------------------- client
class BudgetExceeded(RuntimeError):
    pass


def load_env(path: Path) -> None:
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


class ModelClient:
    """Calls the provided endpoint.  EVERY http attempt is counted against the
    item's budget (retries included), because the grader's proxy counts requests,
    not successes.  Sampling settings are fixed by the brief."""

    def __init__(self, env_path: Path):
        load_env(env_path)
        self.key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.model = os.environ.get("MODEL", "ibm-granite/granite-4.2-8b")
        if not self.key:
            raise RuntimeError("API key missing (expected OPENROUTER_API_KEY in .env)")
        self._lock = threading.Lock()
        self.total_calls = 0
        self.raw = {}

    def complete(self, item_id: str, messages: List[dict], max_tokens: int = 1200) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 1.0, "top_p": 0.95,
                "max_tokens": max_tokens, "reasoning": {"enabled": False}}
        req = urllib.request.Request(
            self.base + "/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                     "X-Item-Id": item_id})
        with self._lock:
            self.total_calls += 1
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read().decode())
        txt = payload["choices"][0]["message"]["content"] or ""
        with self._lock:
            self.raw.setdefault(item_id, []).append(txt)
        return txt


class ItemBudget:
    """Per-item call allowance.  `spend()` raises before a call that would
    exceed the cap, so the cap holds even if a retry path is added later."""

    def __init__(self, cap: int):
        self.cap, self.used = cap, 0

    def left(self) -> int:
        return self.cap - self.used

    def spend(self) -> None:
        if self.used >= self.cap:
            raise BudgetExceeded()
        self.used += 1


def ask(client, budget: ItemBudget, item_id: str, header: Header, w: World,
        lines: Sequence[str], numbers: Optional[Sequence[int]] = None) -> Optional[Dict[int, Optional[tuple]]]:
    """One model call -> parsed per-line constraints, or None if the call failed.
    A failed attempt still consumed budget (it may have reached the proxy)."""
    nums = list(numbers) if numbers is not None else list(range(1, len(lines) + 1))
    budget.spend()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(header, lines, nums)}]
    try:
        text = client.complete(item_id, msgs)
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError):
        return None
    return parse_output(text, header, w, nums)