#!/usr/bin/env python
"""Experiment 1: controlled perturbations.

Make the perturbed set from the reference utterances in one or more pairs
files, then score it with every scorer:

    python scripts/perturbations.py make \
        --pairs results/notsofar1_dev/pyannote_whisper/pairs.json \
        --out results/perturb/perturbed.jsonl [--paraphrases para.jsonl]
    python scripts/perturbations.py score \
        --set results/perturb/perturbed.jsonl --out results/perturb

``score`` writes auc.json (AUC with bootstrap CI per scorer, plus mean d
per edit type), d.jsonl (every item's d per scorer) and boxplots.png.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from semscore.analysis import auc, bootstrap_auc_ci  # noqa: E402
from semscore.pairs import load_pairs  # noqa: E402
from semscore.perturb import load_paraphrases, make_perturbations  # noqa: E402
from semscore.scorers import PRIORITY, REGISTRY, get_scorer  # noqa: E402


def cmd_make(a):
    utts = []
    for path in a.pairs:
        utts += [r for r in load_pairs(path)["records"] if r["kind"] == "utt"]
    para = load_paraphrases(a.paraphrases) if a.paraphrases else None
    rows = make_perturbations(utts, n=a.n, min_words=a.min_words, seed=a.seed,
                              balance=not a.no_balance, paraphrases=para)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    count = defaultdict(int)
    for r in rows:
        count[(r["class"], r["edit"])] += 1
    print(f"{len(rows)} items from {len({r['src'] for r in rows})} source utterances")
    for k in sorted(count):
        print(f"  {k[0]:10s} {k[1]:12s} {count[k]}")


def cmd_score(a):
    with open(a.set) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    names = a.scorers or PRIORITY
    if names == ["all"]:
        names = list(REGISTRY)
    labels = [r["label"] for r in rows]
    groups = [r["src"] for r in rows]
    d = {}
    summary = {}
    for name in names:
        s = get_scorer(name)
        d[name] = s.dissim_batch([r["ref"] for r in rows], [r["hyp"] for r in rows],
                                 [r.get("context") for r in rows])
        lo, hi = bootstrap_auc_ci(d[name], labels, groups, n_boot=a.n_boot)
        by_edit = defaultdict(list)
        for r, x in zip(rows, d[name]):
            by_edit[r["edit"]].append(x)
        summary[name] = {"auc": auc(d[name], labels), "ci95": [lo, hi],
                         "mean_d_by_edit": {k: sum(v) / len(v)
                                            for k, v in sorted(by_edit.items())},
                         "config": s.config()}
        print(f"{name:16s} AUC={summary[name]['auc']:.3f}  [{lo:.3f}, {hi:.3f}]")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "auc.json"), "w") as f:
        json.dump({"n_items": len(rows), "scorers": summary}, f, indent=1)
    with open(os.path.join(a.out, "d.jsonl"), "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": r["id"], "edit": r["edit"], "label": r["label"],
                                **{n: d[n][i] for n in names}}) + "\n")
    boxplots(rows, d, os.path.join(a.out, "boxplots.png"))


def boxplots(rows, d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edits = sorted({r["edit"] for r in rows}, key=lambda e: (
        -next(r["label"] for r in rows if r["edit"] == e), e))
    fig, axes = plt.subplots(1, len(d), figsize=(3.2 * len(d), 3.4), sharey=False,
                             squeeze=False)
    for ax, (name, xs) in zip(axes[0], d.items()):
        data = [[x for r, x in zip(rows, xs) if r["edit"] == e] for e in edits]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, e in zip(bp["boxes"], edits):
            changing = next(r["label"] for r in rows if r["edit"] == e)
            patch.set_facecolor("#d9822b" if changing else "#4c78a8")
        ax.set_xticks(range(1, len(edits) + 1), edits, rotation=60, fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("d" if ax is axes[0][0] else "")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--pairs", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--n", type=int, default=500)
    m.add_argument("--min-words", type=int, default=5)
    m.add_argument("--paraphrases")
    m.add_argument("--no-balance", action="store_true")
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--set", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--scorers", nargs="+")
    s.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args(argv)
    random.seed(0)
    cmd_make(a) if a.cmd == "make" else cmd_score(a)


if __name__ == "__main__":
    main()
