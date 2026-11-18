"""Compute the tcpSemER alignment once and save the aligned pairs.

Every scorer reads the saved pairs, so speaker assignment and time-constrained
alignment are never recomputed. The alignment is exactly the one used by
:func:`tcpsemer.tcpsemer` (it calls the same planning code), so plugging the
original tcpSemER distance back in reproduces its score.

A pairs file is JSON::

    {
      "meta": {...settings...},
      "sessions": {session_id: {"length": N_ref, "unassigned_hyp_words": n}},
      "records": [ {...}, ... ]
    }

Each record has ``kind``:

* ``"utt"``: a reference utterance and the hypothesis words aligned to it
  (matched, substituted and attached insertions). ``hyp_text`` may be empty
  (all deletions).
* ``"insertion"``: a run of inserted hypothesis words that falls outside
  every widened reference utterance. ``ref_text`` is empty.
* ``"missed"``: a reference speaker with no assigned hypothesis speaker.
  ``hyp_text`` is empty.

and the fields ``session_id, speaker, start_time, end_time, ref_text,
hyp_text, n_ref, n_hyp, overlap, overlap_frac, context``.

``overlap`` is true when the record's time span overlaps a reference segment
of another speaker; ``overlap_frac`` is the overlapped fraction of the span.
``context`` holds up to ``context_turns`` previous reference turns (segments
of any speaker whose start time is before the record's start time, in time
order, oldest first).
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

from tcpsemer.core import _load, _plans  # same alignment as tcpsemer()

PAIRS_VERSION = 1


def _overlap_frac(st: float, et: float, others: List[tuple]) -> float:
    """Fraction of [st, et] covered by the union of the intervals in others."""
    if not (et > st):
        return 0.0
    ivs = sorted((max(a, st), min(b, et)) for a, b in others if b > st and a < et)
    covered, cur_a, cur_b = 0.0, None, None
    for a, b in ivs:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                covered += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        covered += cur_b - cur_a
    return covered / (et - st)


def extract_pairs(reference, hypothesis, *, collar: float = 5.0,
                  regex: Optional[str] = None, uem=None,
                  normalizer: Optional[str] = None, max_gap: float = 2.0,
                  context_turns: int = 3, backend: str = "auto") -> Dict:
    """Align reference and hypothesis and return the pairs dictionary."""
    ref, hyp = _load(reference, hypothesis, regex, uem, normalizer)
    ref_by_sess = ref.groupby("session_id")
    hyp_by_sess = hyp.groupby("session_id")
    plans = _plans(ref_by_sess, hyp_by_sess, collar, max_gap, backend)

    sessions, records = {}, []
    for sess, (length, fixed, utts, runs, missed) in plans.items():
        segs = sorted(
            ({"speaker": str(s["speaker"]), "start_time": float(s["start_time"]),
              "end_time": float(s["end_time"]), "words": s["words"]}
             for s in ref_by_sess[sess] if s["words"].strip()),
            key=lambda s: (s["start_time"], s["end_time"]))

        def context(st):
            prev = [s["words"] for s in segs if s["start_time"] < st]
            return prev[-context_turns:] if context_turns > 0 else []

        def overlap(spk, st, et):
            others = [(s["start_time"], s["end_time"]) for s in segs
                      if s["speaker"] != spk]
            f = _overlap_frac(st, et, others)
            return f > 0, f

        n_runs = sum(n for _, _, n, _, _ in runs)
        n_missed = sum(n for _, n in missed)
        sessions[sess] = {"length": int(length),
                          "unassigned_hyp_words": int(fixed - n_runs - n_missed)}

        for spk, r, h, n_ref, n_hyp, st, et in utts:
            ov, of = overlap(spk, st, et)
            records.append(dict(
                session_id=sess, kind="utt", speaker=spk, start_time=st,
                end_time=et, ref_text=r, hyp_text=h, n_ref=int(n_ref),
                n_hyp=int(n_hyp), overlap=ov, overlap_frac=of,
                context=context(st)))
        for spk, h, n, st, et in runs:
            ov, of = overlap(spk, st, et)
            records.append(dict(
                session_id=sess, kind="insertion", speaker=spk, start_time=st,
                end_time=et, ref_text="", hyp_text=h, n_ref=0, n_hyp=int(n),
                overlap=ov, overlap_frac=of, context=context(st)))
        for spk, n in missed:
            text = " ".join(s["words"] for s in segs if s["speaker"] == spk)
            records.append(dict(
                session_id=sess, kind="missed", speaker=spk,
                start_time=math.nan, end_time=math.nan, ref_text=text,
                hyp_text="", n_ref=int(n), n_hyp=0, overlap=None,
                overlap_frac=None, context=[]))

    meta = dict(version=PAIRS_VERSION, collar=collar, max_gap=max_gap,
                normalizer=normalizer, regex=regex,
                uem=uem if isinstance(uem, (str, type(None))) else "<object>",
                context_turns=context_turns,
                reference=reference if isinstance(reference, str) else "<object>",
                hypothesis=hypothesis if isinstance(hypothesis, str) else "<object>")
    return {"meta": meta, "sessions": sessions, "records": records}


def save_pairs(pairs: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(pairs, f, indent=1, allow_nan=True)


def load_pairs(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)
