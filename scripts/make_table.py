#!/usr/bin/env python
"""Main results table: cpWER / tcpWER / each scorer, per corpus and system.

    python scripts/make_table.py [results] [--latex]
"""
import argparse
import glob
import json
import os


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="results")
    p.add_argument("--latex", action="store_true")
    a = p.parse_args(argv)

    rows, scorers = [], []
    for base in sorted(glob.glob(os.path.join(a.root, "*", "*", "baselines.json"))):
        folder = os.path.dirname(base)
        system = os.path.basename(folder)
        corpus = os.path.basename(os.path.dirname(folder))
        with open(base) as f:
            b = json.load(f)
        row = {"corpus": corpus, "system": system,
               "cpWER": b["cpWER"]["error_rate"], "tcpWER": b["tcpWER"]["error_rate"]}
        for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
            name = os.path.splitext(os.path.basename(path))[0]
            if name in ("pairs", "baselines", "agreement"):
                continue
            with open(path) as f:
                row[name] = json.load(f)["corpus"]["error_rate"]
            if name not in scorers:
                scorers.append(name)
        rows.append(row)
    cols = ["cpWER", "tcpWER"] + scorers

    def fmt(x):
        return "–" if x is None else f"{100 * x:.1f}"

    if a.latex:
        print("\\begin{tabular}{ll" + "r" * len(cols) + "}\n\\toprule")
        print("Corpus & System & " + " & ".join(cols) + " \\\\\n\\midrule")
        for r in rows:
            print(f"{r['corpus']} & {r['system']} & "
                  + " & ".join(fmt(r.get(c)) for c in cols) + " \\\\")
        print("\\bottomrule\n\\end{tabular}")
    else:
        print("| corpus | system | " + " | ".join(cols) + " |")
        print("|---|---|" + "---:|" * len(cols))
        for r in rows:
            print(f"| {r['corpus']} | {r['system']} | "
                  + " | ".join(fmt(r.get(c)) for c in cols) + " |")


if __name__ == "__main__":
    main()
