#!/usr/bin/env python
"""Experiment 2: agreement between scorers on real system outputs.

    python scripts/agreement.py results/notsofar1_dev/pyannote_whisper \
        [--a wer --b semdist-minilm --top 20]

Reads every {scorer}.json in the folder (all must share the same pairs.json)
and writes agreement.json (segment- and recording-level Pearson / Kendall,
overall and split by overlap), heatmap.png and disagreements.jsonl.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from semscore.analysis import correlation_matrix, correlations, top_disagreements  # noqa: E402
from semscore.pairs import load_pairs  # noqa: E402


def _key(k):
    return f"{k[0]}~{k[1]}"


def heatmap(names, m, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(names), 0.8 + 0.6 * len(names)))
    im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            v = m[i][j]
            ax.text(j, i, "–" if v != v else f"{v:.2f}", ha="center",
                    va="center", fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--a", default="wer", help="n-gram/edit scorer for disagreements")
    p.add_argument("--b", default="semdist-minilm", help="embedding scorer")
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args(argv)

    res = {}
    for path in sorted(glob.glob(os.path.join(a.folder, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in ("pairs", "baselines", "agreement"):
            continue
        with open(path) as f:
            res[name] = json.load(f)
    pairs = load_pairs(os.path.join(a.folder, "pairs.json"))
    recs = pairs["records"]
    utt = [i for i, r in enumerate(recs) if r["kind"] == "utt" and r["n_hyp"] > 0]

    def seg_table(idx):
        return {n: [r["d"][i] for i in idx] for n, r in res.items()}

    out = {"n_scored_utts": len(utt), "segment": {}, "recording": {}}
    for region, idx in [("all", utt),
                        ("overlap", [i for i in utt if recs[i]["overlap"]]),
                        ("non_overlap", [i for i in utt if not recs[i]["overlap"]])]:
        out["segment"][region] = {_key(k): v for k, v in
                                  correlations(seg_table(idx)).items()}
    sessions = sorted(pairs["sessions"])
    rec_table = {n: [r["sessions"][s]["error_rate"] for s in sessions]
                 for n, r in res.items()}
    out["recording"]["all"] = {_key(k): v for k, v in correlations(rec_table).items()}

    with open(os.path.join(a.folder, "agreement.json"), "w") as f:
        json.dump(out, f, indent=1)
    names, m = correlation_matrix(seg_table(utt), "kendall")
    heatmap(names, m, os.path.join(a.folder, "heatmap.png"),
            "Segment-level Kendall τ")
    if a.a in res and a.b in res:
        rows = top_disagreements(recs, res[a.a]["d"], res[a.b]["d"], a.top)
        with open(os.path.join(a.folder, "disagreements.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"top {len(rows)} {a.a} vs {a.b} disagreements -> disagreements.jsonl")
    for k, v in out["segment"]["all"].items():
        print(f"{k:36s} r={v['pearson']:.3f} tau={v['kendall']:.3f} n={v['n']}")


if __name__ == "__main__":
    main()
