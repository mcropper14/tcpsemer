"""Rule-based meaning-changing and meaning-preserving edits (Experiment 1).

Each edit function takes a list of words and a ``random.Random`` and returns
the edited word list, or ``None`` if the edit does not apply to that
utterance. Paraphrases cannot be made by rules; they are read from a file
(hand-written or LLM-generated and checked by a person), see
:func:`load_paraphrases`.

Inputs are expected in the normalised style of the references (lower case,
no punctuation), e.g. after the ``chime8`` normaliser.
"""
from __future__ import annotations

import json
import random
from typing import Callable, Dict, List, Optional

from .scorers import word_edit_distance

Words = List[str]

# ---- word lists -----------------------------------------------------------

AUX = ["is", "are", "was", "were", "do", "does", "did", "can", "could",
       "will", "would", "should", "have", "has", "had", "must"]
NEG_CONTRACTED = {"isn't": "is", "aren't": "are", "wasn't": "was",
                  "weren't": "were", "don't": "do", "doesn't": "does",
                  "didn't": "did", "can't": "can", "couldn't": "could",
                  "won't": "will", "wouldn't": "would", "shouldn't": "should",
                  "haven't": "have", "hasn't": "has", "hadn't": "had",
                  "mustn't": "must"}
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
           "eight", "nine", "ten", "eleven", "twelve", "fifteen", "twenty",
           "thirty", "forty", "fifty", "hundred", "thousand", "million"]
NAMES = ["john", "mary", "david", "sarah", "michael", "lisa", "james", "anna",
         "peter", "emma", "paul", "kate", "tom", "laura", "mark", "susan",
         "london", "paris", "tokyo", "berlin", "monday", "tuesday",
         "wednesday", "thursday", "friday", "saturday", "sunday", "january",
         "february", "march", "april", "june", "july", "august",
         "september", "october", "november", "december"]
ANTONYM_PAIRS = [("yes", "no"), ("before", "after"), ("always", "never"),
                 ("more", "less"), ("up", "down"), ("left", "right"),
                 ("open", "close"), ("start", "stop"), ("good", "bad"),
                 ("agree", "disagree"), ("true", "false"),
                 ("increase", "decrease"), ("early", "late"),
                 ("first", "last"), ("on", "off"), ("accept", "reject"),
                 ("buy", "sell"), ("high", "low"), ("big", "small"),
                 ("hot", "cold"), ("easy", "hard"), ("cheap", "expensive"),
                 ("win", "lose"), ("everyone", "nobody"),
                 ("everything", "nothing"), ("possible", "impossible"),
                 ("better", "worse"), ("best", "worst"), ("above", "below")]
ANTONYMS = {**{a: b for a, b in ANTONYM_PAIRS}, **{b: a for a, b in ANTONYM_PAIRS}}
FILLERS = ["uh", "um", "like", "you know"]
FILLERS_DELETABLE = {"uh", "um", "uhm", "umm", "hmm", "er", "erm", "ah"}
CONTRACTIONS = {("do", "not"): "don't", ("does", "not"): "doesn't",
                ("did", "not"): "didn't", ("is", "not"): "isn't",
                ("are", "not"): "aren't", ("was", "not"): "wasn't",
                ("can", "not"): "can't", ("will", "not"): "won't",
                ("i", "am"): "i'm", ("you", "are"): "you're",
                ("we", "are"): "we're", ("they", "are"): "they're",
                ("it", "is"): "it's", ("that", "is"): "that's",
                ("there", "is"): "there's", ("i", "will"): "i'll",
                ("we", "will"): "we'll", ("i", "have"): "i've",
                ("we", "have"): "we've", ("let", "us"): "let's",
                ("i", "would"): "i'd", ("what", "is"): "what's"}
UNCONTRACT = {v: list(k) for k, v in CONTRACTIONS.items()}


# ---- meaning-changing -----------------------------------------------------

def negation(w: Words, rng: random.Random) -> Optional[Words]:
    """Delete a negation (``not``, ``isn't`` -> ``is``) or insert ``not``
    after an auxiliary."""
    dels = [i for i, x in enumerate(w) if x == "not" or x in NEG_CONTRACTED]
    ins = [i for i, x in enumerate(w) if x in AUX
           and (i + 1 >= len(w) or w[i + 1] != "not")]
    opts = [("del", i) for i in dels] + [("ins", i) for i in ins]
    if not opts:
        return None
    op, i = rng.choice(opts)
    if op == "del":
        return w[:i] + w[i + 1:] if w[i] == "not" else w[:i] + [NEG_CONTRACTED[w[i]]] + w[i + 1:]
    return w[:i + 1] + ["not"] + w[i + 1:]


def _substitute(w, rng, vocab, choices):
    idx = [i for i, x in enumerate(w) if x in vocab]
    if not idx:
        return None
    i = rng.choice(idx)
    return w[:i] + [rng.choice([c for c in choices(w[i]) if c != w[i]])] + w[i + 1:]


def number(w, rng):
    digit = [i for i, x in enumerate(w) if x.isdigit()]
    if digit and (rng.random() < 0.5 or not any(x in NUMBERS for x in w)):
        i = rng.choice(digit)
        new = str(int(w[i]) + rng.choice([-2, -1, 1, 2, 10]))
        new = new if new != w[i] and not new.startswith("-") else str(int(w[i]) + 1)
        return w[:i] + [new] + w[i + 1:]
    return _substitute(w, rng, set(NUMBERS), lambda x: NUMBERS)


def entity(w, rng):
    def same_kind(x):
        k = NAMES.index(x)
        # keep people / places / weekdays / months apart
        groups = [(0, 16), (16, 20), (20, 27), (27, len(NAMES))]
        a, b = next(g for g in groups if g[0] <= k < g[1])
        return NAMES[a:b]
    return _substitute(w, rng, set(NAMES), same_kind)


def antonym(w, rng):
    return _substitute(w, rng, set(ANTONYMS), lambda x: [ANTONYMS[x]])


# ---- meaning-preserving ---------------------------------------------------

def filler(w, rng):
    """Delete an existing filler if there is one, else insert one."""
    dels = [i for i, x in enumerate(w) if x in FILLERS_DELETABLE]
    if dels and rng.random() < 0.5:
        i = rng.choice(dels)
        return w[:i] + w[i + 1:]
    i = rng.randrange(1, len(w)) if len(w) > 1 else 1
    return w[:i] + rng.choice(FILLERS).split() + w[i:]


def contraction(w, rng):
    opts = [("c", i) for i in range(len(w) - 1) if (w[i], w[i + 1]) in CONTRACTIONS]
    opts += [("u", i) for i, x in enumerate(w) if x in UNCONTRACT]
    if not opts:
        return None
    op, i = rng.choice(opts)
    if op == "c":
        return w[:i] + [CONTRACTIONS[(w[i], w[i + 1])]] + w[i + 2:]
    return w[:i] + UNCONTRACT[w[i]] + w[i + 1:]


def repetition(w, rng):
    """Repeat a word or a two-word span (a stutter / restart)."""
    i = rng.randrange(len(w))
    n = 2 if i + 1 < len(w) and rng.random() < 0.3 else 1
    return w[:i + n] + w[i:i + n] + w[i + n:]


CHANGING: Dict[str, Callable] = {"negation": negation, "number": number,
                                 "entity": entity, "antonym": antonym}
PRESERVING: Dict[str, Callable] = {"filler": filler, "contraction": contraction,
                                   "repetition": repetition}


def load_paraphrases(path: str) -> Dict[str, str]:
    """JSON lines with ``{"ref": ..., "paraphrase": ...}`` -> dict."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["ref"]] = d["paraphrase"]
    return out


def make_perturbations(utterances: List[Dict], n: int = 500, min_words: int = 5,
                       seed: int = 0, balance: bool = True,
                       paraphrases: Optional[Dict[str, str]] = None) -> List[Dict]:
    """Build the perturbed set.

    ``utterances``: dicts with at least ``ref_text`` (and optionally
    ``context``, ``session_id``, ``speaker``, ``start_time``). ``n`` source
    utterances with at least ``min_words`` words are sampled, every
    applicable edit is applied to each, and, if ``balance``, the two classes
    are subsampled so that every word-edit-distance bucket has the same number
    of items in both classes (so WER alone cannot separate them).
    """
    rng = random.Random(seed)
    pool = [u for u in utterances if len(u["ref_text"].split()) >= min_words]
    seen, uniq = set(), []
    for u in pool:
        if u["ref_text"] not in seen:
            seen.add(u["ref_text"])
            uniq.append(u)
    src = rng.sample(uniq, min(n, len(uniq)))

    items = []
    for si, u in enumerate(src):
        w = u["ref_text"].split()
        edits = [(1, k, f) for k, f in CHANGING.items()] + \
                [(0, k, f) for k, f in PRESERVING.items()]
        for label, kind, f in edits:
            out = f(list(w), rng)
            if out is None or out == w:
                continue
            items.append((si, label, kind, " ".join(out)))
        if paraphrases and u["ref_text"] in paraphrases:
            items.append((si, 0, "paraphrase", paraphrases[u["ref_text"]]))

    rows = []
    for si, label, kind, hyp in items:
        u = src[si]
        ref = u["ref_text"]
        rows.append({"src": si, "ref": ref, "hyp": hyp,
                     "label": label,  # 1 = meaning-changing
                     "class": "changing" if label else "preserving",
                     "edit": kind,
                     "n_edit": word_edit_distance(ref.split(), hyp.split()),
                     "context": u.get("context", []),
                     "session_id": u.get("session_id"),
                     "speaker": u.get("speaker"),
                     "start_time": u.get("start_time")})

    if balance:
        by = {}
        for r in rows:
            by.setdefault((r["n_edit"], r["label"]), []).append(r)
        kept = []
        for e in sorted({k[0] for k in by}):
            a, b = by.get((e, 0), []), by.get((e, 1), [])
            m = min(len(a), len(b))
            kept += rng.sample(a, m) + rng.sample(b, m)
        rows = sorted(kept, key=lambda r: (r["src"], r["label"], r["edit"]))
    for i, r in enumerate(rows):
        r["id"] = i
    return rows
