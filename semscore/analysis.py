"""Analysis helpers for Experiments 1 and 2."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional


def auc(scores: List[float], labels: List[int]) -> float:
    """ROC AUC of ``scores`` for ``labels`` (1 = meaning-changing should get the
    larger distance). Ties count one half."""
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return math.nan
    return float(roc_auc_score(labels, scores))


def bootstrap_auc_ci(scores, labels, groups=None, n_boot: int = 1000,
                     seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the AUC. If ``groups`` (source utterance ids)
    is given, resample source utterances rather than items, since edits of the
    same utterance are not independent."""
    import numpy as np
    rng = np.random.default_rng(seed)
    scores, labels = np.asarray(scores), np.asarray(labels)
    if groups is None:
        groups = np.arange(len(scores))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        a = auc(scores[idx].tolist(), labels[idx].tolist())
        if not math.isnan(a):
            vals.append(a)
    if not vals:
        return math.nan, math.nan
    return (float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


def correlations(table: Dict[str, List[Optional[float]]]) -> Dict:
    """Pearson r and Kendall tau between every pair of columns, using rows
    where both are defined."""
    from scipy.stats import kendalltau, pearsonr
    out = {}
    for a, b in combinations(table, 2):
        xs = [(x, y) for x, y in zip(table[a], table[b])
              if x is not None and y is not None
              and not (isinstance(x, float) and math.isnan(x))
              and not (isinstance(y, float) and math.isnan(y))]
        if len(xs) < 3:
            out[(a, b)] = {"pearson": math.nan, "kendall": math.nan, "n": len(xs)}
            continue
        x, y = zip(*xs)
        const = len(set(x)) == 1 or len(set(y)) == 1
        out[(a, b)] = {
            "pearson": math.nan if const else float(pearsonr(x, y)[0]),
            "kendall": math.nan if const else float(kendalltau(x, y)[0]),
            "n": len(xs)}
    return out


def correlation_matrix(table, method: str = "kendall"):
    names = list(table)
    corr = correlations(table)
    m = [[1.0 if a == b else corr.get((a, b), corr.get((b, a)))[method]
          for b in names] for a in names]
    return names, m


def top_disagreements(records: List[Dict], d_a: List[Optional[float]],
                      d_b: List[Optional[float]], n: int = 20) -> List[Dict]:
    """Records with the largest |d_a - d_b| (only scored utterances)."""
    rows = [(abs(a - b), i, a, b) for i, (a, b) in enumerate(zip(d_a, d_b))
            if a is not None and b is not None
            and records[i]["kind"] == "utt" and records[i]["n_hyp"] > 0]
    rows.sort(reverse=True)
    out = []
    for gap, i, a, b in rows[:n]:
        r = records[i]
        out.append({"index": i, "gap": gap, "d_a": a, "d_b": b,
                    "session_id": r["session_id"], "speaker": r["speaker"],
                    "start_time": r["start_time"], "ref": r["ref_text"],
                    "hyp": r["hyp_text"], "overlap": r["overlap"]})
    return out
