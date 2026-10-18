"""DEV ONLY.  Template-regex labeller for the *visible* set.

Purpose: (1) confirm my reading of every phrasing against visible_key.json,
(2) produce gold line-level labels to score the model's extraction.
It is deliberately NOT used by ./run -- regexes over known templates are exactly
the thing that breaks on the held-out phrasing pool.
"""
import re

HEDGES = ["I'm fairly sure ", "From what I recall, ", "As far as I know, ", "I believe ",
          "Going off the roster, ", "If memory serves, ", "My recollection is that ",
          "Unless I have this wrong, ", "Speaking from memory, ", "I'm reasonably confident "]
RESTATE = re.compile(r"^(Noted twice in the handover|This came up more than once, so|It bears repeating|"
                     r"Worth restating, since it came up twice in handover): ")


def strip_prefixes(line):
    s = line
    for _ in range(3):
        s = RESTATE.sub("", s)
        for h in HEDGES:
            if s.startswith(h):
                s = s[len(h):]
        if s and s[0].islower() and s.split()[0] == "put":
            pass
    return s


def label(line, h):
    """-> (op, args) or None"""
    s = strip_prefixes(line).rstrip(".")
    N = r"(" + "|".join(map(re.escape, h.people)) + r")"
    T = r"(\d\d:\d\d)"
    S = r"(" + "|".join(h.stations) + r")"
    Sc = r"(" + "|".join(x.capitalize() for x in h.stations) + r"|" + "|".join(h.stations) + r")"
    low = lambda x: x.lower()
    pats = [
        # AT
        (rf"^{N} takes {T}, as things stand$", lambda m: ("AT", (m[1], m[2]))),
        (rf"^{T} is when {N} is scheduled$", lambda m: ("AT", (m[2], m[1]))),
        (rf"^[Tt]he {T} block is {N}'s$", lambda m: ("AT", (m[2], m[1]))),
        (rf"^[Ww]hoever drew the {T} block, it was {N}$", lambda m: ("AT", (m[2], m[1]))),
        (rf"^{N} is on the {T} block$", lambda m: ("AT", (m[1], m[2]))),
        (rf"^{N} is the one who opens up at {T}$", lambda m: ("AT", (m[1], m[2]))),
        # NOT_AT
        (rf"^{N} is not on the {T} block$", lambda m: ("NOT_AT", (m[1], m[2]))),
        (rf"^[Tt]he {T} block is not {N}'s$", lambda m: ("NOT_AT", (m[2], m[1]))),
        (rf"^{N} is unavailable at {T}$", lambda m: ("NOT_AT", (m[1], m[2]))),
        (rf"^{N} has a standing commitment that rules out the {T} block entirely$", lambda m: ("NOT_AT", (m[1], m[2]))),
        (rf"^[Yy]ou will not find {N} on the {T} block$", lambda m: ("NOT_AT", (m[1], m[2]))),
        (rf"^{T} is the one block {N} is definitely not on$", lambda m: ("NOT_AT", (m[2], m[1]))),
        # ON
        (rf"^{Sc} is covered by {N}$", lambda m: ("ON", (m[1].lower(), m[2]))),
        (rf"^[Tt]he {S} station is down to {N}$", lambda m: ("ON", (m[1], m[2]))),
        (rf"^{N} has {S} this week$", lambda m: ("ON", (m[2], m[1]))),
        (rf"^{N} is assigned to {S}$", lambda m: ("ON", (m[2], m[1]))),
        (rf"^{N} is on {S}$", lambda m: ("ON", (m[2], m[1]))),
        (rf"^{Sc} is {N}'s station$", lambda m: ("ON", (m[1].lower(), m[2]))),
        # NOT_ON
        (rf"^{Sc} is covered by someone other than {N}$", lambda m: ("NOT_ON", (m[1].lower(), m[2]))),
        (rf"^{N} is not on {S}$", lambda m: ("NOT_ON", (m[2], m[1]))),
        (rf"^{N} is not assigned to {S}$", lambda m: ("NOT_ON", (m[2], m[1]))),
        (rf"^{N} has not been put on {S}$", lambda m: ("NOT_ON", (m[2], m[1]))),
        (rf"^[Yy]ou can rule {N} out for {S}$", lambda m: ("NOT_ON", (m[2], m[1]))),
        (rf"^{Sc} is not {N}'s station$", lambda m: ("NOT_ON", (m[1].lower(), m[2]))),
        # BEFORE
        (rf"^{N} works earlier in the day than {N}$", lambda m: ("BEFORE", (m[1], m[2]))),
        (rf"^{N} comes later in the day than {N}$", lambda m: ("BEFORE", (m[2], m[1]))),
        (rf"^{N}'s block falls somewhere earlier than {N}'s$", lambda m: ("BEFORE", (m[1], m[2]))),
        (rf"^[Bb]y the time {N} starts, {N} has already been on$", lambda m: ("BEFORE", (m[2], m[1]))),
        (rf"^{N} is done before {N} starts$", lambda m: ("BEFORE", (m[1], m[2]))),
        (rf"^{N} takes over from {N} later in the day$", lambda m: ("BEFORE", (m[2], m[1]))),
        # NEXT
        (rf"^{N} is on the block directly after {N}$", lambda m: ("NEXT", (m[2], m[1]))),
        (rf"^{N} relieves {N} directly, with no block in between$", lambda m: ("NEXT", (m[2], m[1]))),
        (rf"^{N} hands straight over to {N}$", lambda m: ("NEXT", (m[1], m[2]))),
        (rf"^{N} then {N}, back to back$", lambda m: ("NEXT", (m[1], m[2]))),
        (rf"^{N} is on the block immediately before {N}$", lambda m: ("NEXT", (m[1], m[2]))),
        (rf"^[Tt]here is no block between {N}'s and {N}'s, in that order$", lambda m: ("NEXT", (m[1], m[2]))),
        # BETWEEN
        (rf"^{N} sits between {N} and {N} on the rota$", lambda m: ("BETWEEN", (m[1], m[2], m[3]))),
        (rf"^[Ww]hichever way round {N} and {N} are, {N} is between them$", lambda m: ("BETWEEN", (m[3], m[1], m[2]))),
        (rf"^{N} works at some point between {N} and {N}, not necessarily adjacent to either$", lambda m: ("BETWEEN", (m[1], m[2], m[3]))),
        (rf"^[Pp]ut {N} between {N} and {N}, though not necessarily next to either$", lambda m: ("BETWEEN", (m[1], m[2], m[3]))),
        (rf"^{N} is on after one of {N} and {N} and before the other$", lambda m: ("BETWEEN", (m[1], m[2], m[3]))),
        (rf"^{N}'s block falls between {N}'s and {N}'s, in one order or the other$", lambda m: ("BETWEEN", (m[1], m[2], m[3]))),
        # S_BEFORE / S_AFTER
        (rf"^[Tt]he {S} station is covered earlier in the day than {N}'s block$", lambda m: ("S_BEFORE", (m[1], m[2]))),
        (rf"^[Ww]hoever is on {S} works earlier in the day than {N}$", lambda m: ("S_BEFORE", (m[1], m[2]))),
        (rf"^[Tt]he person on {S} precedes {N}$", lambda m: ("S_BEFORE", (m[1], m[2]))),
        (rf"^[Ww]hoever has {S} is done before {N} starts$", lambda m: ("S_BEFORE", (m[1], m[2]))),
        (rf"^{Sc} is covered before {N} comes on$", lambda m: ("S_BEFORE", (m[1].lower(), m[2]))),
        (rf"^{N} is on later than whoever has {S}$", lambda m: ("S_BEFORE", (m[2], m[1]))),
    ]
    for p, f in pats:
        m = re.match(p, s)
        if m:
            r = f(m)
            if r and r[0] in ("ON", "NOT_ON"):
                r = (r[0], (r[1][1], r[1][0]))   # patterns emit (station, person)
            return r
    return None
