"""DEV ONLY: a stand-in 'model' for offline plumbing tests.  It answers from the
regex oracle and corrupts each line with probability `noise`, rendering the same
text format a real model would.  Counts calls per item like the grader's proxy."""
import random, re, collections, threading
from auric.core import parse_header, split_item
from auric.extract import OPS
from dev.oracle import label


class FakeClient:
    def __init__(self, noise=0.1, seed=0):
        self.noise, self.rng = noise, random.Random(seed)
        self.calls = collections.Counter(); self.total_calls = 0
        self._l = threading.Lock()

    def complete(self, item_id, messages, max_tokens=0):
        with self._l:
            self.calls[item_id] += 1; self.total_calls += 1
        user = messages[-1]["content"]
        text = user.split("NOTES:\n")[-1].split("\nOUTPUT:")[0]
        rota = re.search(r"ROTA people: (.*?) \| blocks .*?\| stations: (.*?) \| people on a station: (.*)", user.split("Now do the real task")[1])
        from auric.core import Header
        people = tuple(rota.group(1).split(", "))
        h = Header(people, ("07:00","09:00","11:00","13:00","15:00"), tuple(rota.group(2).split(", ")), tuple(rota.group(3).split(", ")))
        out = []
        for ln in text.splitlines():
            m = re.match(r"(\d+)\. (.*)", ln)
            n, t = m.group(1), m.group(2)
            c = label(t, h)
            if self.rng.random() < self.noise:
                c = None if c else ("AT", (self.rng.choice(people), self.rng.choice(h.blocks)))
            out.append(f"{n} | NONE" if c is None else f"{n} | {c[0]} | " + " | ".join(c[1]))
        return "\n".join(out)
