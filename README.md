# Auric AI take-home: weak-model extraction + exact symbolic solving

**Candidate:** Roicy Vinu  |  **Python:** 3.11, standard library only (no third-party deps; see requirements.txt)

## Run
    ./run items.json --budget 1x|3x|10x --out answers.json     (Windows: python run items.json ...)
Needs a `.env` (not committed) with OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL=ibm-granite/granite-4.2-8b.
Every call uses temperature 1.0, top_p 0.95, reasoning disabled, and an `X-Item-Id` header.

## Results (all measured live on granite-4.2-8b; temperature 1.0, top_p 0.95, reasoning off)
Headline = macro exact match over the three kinds (score.py). "Visible" = the 60 provided items; "Shifted" = the same 60 puzzles with every
constraint re-worded from a disjoint template/hedge pool and new distractors (dev/paraphrase.py; my own held-out proxy, not the graders').

| System | Budget | Visible macro | Shifted macro | Line accuracy (visible / shifted) | Runs |
|---|---|---|---|---|---|
| v1 prompt (bare op list, one-line-per-note) | 1x | 16.7, 20.0, 23.3, 18.3 % | not run | 72-80 % / - | 4 |
| v1 prompt | 3x | 25.0 % (unique 20, ambiguous 10, inconsistent 45) | not run | - | 1 |
| **v2 (gloss + wording cheat-sheet + lenient parser)** | **1x** | **76.7 %** (unique 15/20, ambiguous 13/20, inconsistent 19/20) | **45.0 %** (unique 7, ambiguous 16, inconsistent 16 of 20) | 96.4 % / 90.3 % | 1 each |
| v2 | 3x | 73.3 % (unique 13/20, ambiguous 12/20, inconsistent 19/20); 134 calls for 60 items (mean 2.2/item) | not run | - | 1 |
| v2 | 10x | not measured | not measured | - | - |

Caveats, stated plainly: (a) every number is from a single run except where listed, and sampling at temperature 1.0 varies run to run; (b) the v2
cheat-sheet was written after studying the visible wordings, so the *visible* v2 score is optimistic - the shifted column is the honest estimate of
held-out behaviour, and it is much lower; (c) with v2, 3x (73.3%) did not beat 1x (76.7%) - a difference inside run-to-run noise - and v1's 3x gain was only about +5 points, so the measured curve is flat: the remaining errors are systematic (a phrasing the model consistently misreads), which resampling and voting cannot fix. 10x was not measured. I make no claim that extra budget helps; (d) the shifted set is my own construction and may be easier or harder than the graders'.

Ablation (what exists and reproduces): prompt variant `AURIC_VARIANT=v1|v2` (gloss + cheat-sheet vs not); pipeline switches
`python dev/eval.py --budget 3x --ablate vote|adaptive|repair|health|cite_by_conf`. Only the variant comparison above is measured live; the
sampling/voting/repair switches were verified with a simulated model only (plumbing and budget caps), not on Granite.

Per-operator line accuracy, v1 -> v2 (visible, 1x): AT .88->.95, BEFORE .22->.80, BETWEEN .46->.98, NEXT .55->.87, NONE .98->1.00, NOT_AT .72->.94,
NOT_ON .63->.97, ON .82->.97, S_BEFORE .51->.84. On the shifted set v2's weakest are S_BEFORE .62, NOT_AT .76, NEXT .80, AT .81 -> the remaining
errors are the direction/argument-order and re-worded-negation cases.

## Two pages

### 1. How I characterised the model
Granite-4.2-8B scores 0% when asked to solve a puzzle end-to-end. I split the job into what an 8B model can plausibly do (read one sentence, decide what it asserts) and what it cannot (hold a permutation search in its head, count solutions, quote exactly). Items are small: 5 people, 3 stations, so only 5!x3! = 720 candidate rotas. I wrote a regex labeller for the visible phrasings (dev only) and checked that its labels plus an exact solver reproduce all 60 visible answers. That established the ground truth semantics: hedges carry full force, "opens up at 13:00" is a block constraint, imperatives like "Put X between Y and Z" are constraints, historical/hypothetical/wish/unresolved lines and all trivia are not. The first live run (v1 prompt) showed Granite's line-labelling was the bottleneck: ~75% line accuracy sounds fine, but an item is right only if essentially every constraint line is right, and unique/inconsistent items were mostly declared ambiguous (constraints missed). A per-operator diagnostic (dev/eval.py -> diag_*.txt) located three concrete failures: (1) *omissions* - real constraints returned as NONE, partly because the model echoed the sentence before the label and my strict parser dropped it; (2) *direction/argument-order slips* on relational operators (BEFORE 22%, S_BEFORE 51%, e.g. "X is on later than whoever has packing" labelled as the opposite operator); (3) *wishes read as facts* - "wanted 07:00 and did not get it" labelled NOT_AT. Distractor filtering itself was fine (NONE 98%). Resampling (3x) barely helped (+5 points): the errors were systematic, not noise. Fixing the format (a plain-words gloss before the label, a lenient parser) and spelling out each operator's direction took visible 1x from ~20% to 77%.

### 2. Why each component exists
- **Deterministic header/line parsing:** structural, no reason to spend model capacity on it; gives numbered lines so the model only labels.
- **Closed constraint vocabulary (9 types) + one-line-per-note output with a plain-words gloss before the label, and a wording->operator cheat-sheet:** the gloss makes the model read each line explicitly (it cut omissions and direction slips); the cheat-sheet fixes argument order. This is the component that mattered most (about +55 points at 1x). Caveat: it encodes the visible wordings, so it is the part most exposed to new phrasing (shifted: 45%).
- **Lenient parser with type-based swap repair (symbolic):** finds the operator anywhere on the line and takes the next valid names/blocks/stations; swapped station/person or block/person order is fixed deterministically. Part of why v1 lost constraints.
- **(was) Closed vocabulary:** the model never writes free text, never solves; output is parseable and every label is validated (names/blocks/stations must exist; swapped argument order is repaired by type).
- **Exact enumerate-and-filter solver:** decides unique/ambiguous/inconsistent without the model; this is the formal representation that stops "always unique" behaviour and makes the 3x3 matrix meaningful.
- **Minimal conflict sets by subset search, cited by line index:** the cited text is copied from the notes, so it is verbatim by construction; one line per distinct constraint matches the minimal-core rule.
- **Sampling + per-line voting, early stopping, focused re-asks, plausibility check (ambiguous has 2-4 solutions, conflicts have >=3 lines):** use extra budget where the model disagrees with itself. Implemented and unit-tested with a simulated model, **unmeasured on Granite**.
- **Budget accounting:** every HTTP attempt is counted against the per-item cap; every item gets >=1 call; the answer depends on the model output (empty output gives a different answer).

### What I built and removed
My v1 prompt (bare operator list, no gloss, no cheat-sheet) is kept as AURIC_VARIANT=v1 and scored ~20% at 1x; v2 replaced it. My first baseline asked the model to return the whole answer at temperature 0; it violated the required sampling settings, mixed extraction with solving, and was replaced. I also prototyped a z3 solver and dropped it: with 720 candidates brute force is exact, dependency-free, and lets me enumerate every conflict set.

### Where it breaks
(1) Missed or mis-directed constraints at the labelling step; one wrong line flips the kind (shifted set: S_BEFORE .62, NOT_AT .76). (2) The vocabulary is closed: a held-out phrasing that needs a constraint type outside the 9 (e.g. "not adjacent", "exactly two blocks apart") cannot be expressed. (3) The plausibility check assumes the brief's priors. (4) At 1x there is no recourse for a bad sample. (5) Voting only helps if errors are independent across samples; a systematic misreading of a phrasing is repeated every time.

### Extraction: model-based or symbolic? And different authors?
**Constraint extraction is model-based** (Granite labels each line); header parsing, solving, conflict search and citation are symbolic. I deliberately did not ship template regexes: they reproduce the visible set perfectly (dev/oracle.py) and would fail on held-out wording, which is exactly what they would be overfitted to. (The v2 prompt's cheat-sheet is a softer version of the same risk: it lists visible wordings as *examples* for the model to generalise from, and the shifted-set drop shows how much that costs.) If notes were written by different people, the model-based extractor degrades gracefully on paraphrase but the closed vocabulary, informal hedging, pronouns, multi-clause sentences and implied constraints would hurt. I built a phrasing-shifted copy of the visible set (dev/paraphrase.py, new hedges, new templates, new distractors) as a held-out proxy, and ran v2 on it once live: 45% macro vs 77% on the visible wording - so the honest held-out expectation is well below the visible score.

### With a month
Measure per-operator error on the live model and fix the prompt/format against it; per-line isolated calls to see the ceiling of the model on a single sentence; constrained decoding or a label-per-line scoring pass; a learned-free ensemble (several prompt phrasings, not just resamples, to decorrelate errors); a wider vocabulary with a catch-all "unparsed" that triggers a repair loop; a real held-out paraphrase set written by several people.

## Files
`auric/` (core.py solver, extract.py prompt/parser/client, pipeline.py voting/health/answer, cli.py), `run`, `dev/` (oracle, paraphrase, fake model, eval).