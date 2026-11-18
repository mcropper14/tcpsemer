"""Turn per-pair dissimilarities into error rates.

Rules (same as tcpSemER):

* ``utt`` with a non-empty hypothesis: errors = |R| * d, where d comes from the scorer.
* ``utt`` with an empty hypothesis (all deletions): d = 1, errors = |R|.
* ``insertion`` (inserted words outside every widened utterance): errors = |H|.
* ``missed`` reference speaker: errors = |R|.
* Words of hypothesis speakers not assigned to any reference speaker:
  errors = their count (stored per session as ``unassigned_hyp_words``).

The error rate is errors / N_ref (micro average over sessions and corpus).
``mean_d`` is the unweighted mean of d over ``utt`` records (deletions
included with d = 1).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from .scorers import Scorer


def score_records(records: List[Dict], scorer: Scorer) -> List[Optional[float]]:
    """d for every record (None for insertion / missed records)."""
    idx = [i for i, r in enumerate(records)
           if r["kind"] == "utt" and r["n_hyp"] > 0 and r["hyp_text"].strip()]
    ds = scorer.dissim_batch([records[i]["ref_text"] for i in idx],
                             [records[i]["hyp_text"] for i in idx],
                             [records[i].get("context") for i in idx])
    out: List[Optional[float]] = [None] * len(records)
    for r_i, rec in enumerate(records):
        if rec["kind"] == "utt":
            out[r_i] = 1.0  # deletion, overwritten below if scored
    for i, d in zip(idx, ds):
        out[i] = float(d)
    return out


def record_errors(rec: Dict, d: Optional[float]) -> float:
    if rec["kind"] == "utt":
        return rec["n_ref"] * d
    if rec["kind"] == "insertion":
        return float(rec["n_hyp"])
    if rec["kind"] == "missed":
        return float(rec["n_ref"])
    raise ValueError(rec["kind"])


def _summary(errors: float, length: int, ds: List[float]) -> Dict:
    return {"errors": errors, "length": length,
            "error_rate": errors / length if length else math.nan,
            "mean_d": sum(ds) / len(ds) if ds else math.nan,
            "n_utts": len(ds)}


def aggregate(pairs: Dict, d: List[Optional[float]]) -> Dict:
    """Per-session, corpus and overlap / non-overlap summaries."""
    recs = pairs["records"]
    sess = {s: {"errors": float(v["unassigned_hyp_words"]), "length": v["length"],
                "ds": []} for s, v in pairs["sessions"].items()}
    region = {k: {"errors": 0.0, "length": 0, "ds": []}
              for k in ("overlap", "non_overlap")}
    for rec, di in zip(recs, d):
        e = record_errors(rec, di)
        s = sess[rec["session_id"]]
        s["errors"] += e
        if rec["kind"] == "utt":
            s["ds"].append(di)
        if rec["overlap"] is not None:
            g = region["overlap" if rec["overlap"] else "non_overlap"]
            g["errors"] += e
            g["length"] += rec["n_ref"]
            if rec["kind"] == "utt":
                g["ds"].append(di)

    sessions = {k: _summary(v["errors"], v["length"], v["ds"]) for k, v in sess.items()}
    corpus = _summary(sum(v["errors"] for v in sess.values()),
                      sum(v["length"] for v in sess.values()),
                      [x for v in sess.values() for x in v["ds"]])
    corpus["n_sessions"] = len(sessions)
    by_overlap = {k: _summary(v["errors"], v["length"], v["ds"])
                  for k, v in region.items()}
    return {"corpus": corpus, "sessions": sessions, "by_overlap": by_overlap}


def run_scorer(pairs: Dict, scorer: Scorer) -> Dict:
    """Score all pairs and return the full result dictionary."""
    d = score_records(pairs["records"], scorer)
    res = aggregate(pairs, d)
    res["scorer"] = scorer.config()
    res["d"] = d
    return res
