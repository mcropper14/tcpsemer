#!/usr/bin/env python
"""Align once, then run every scorer on one (corpus, system).

    python scripts/run_all.py --corpus notsofar1_dev --system pyannote_whisper
    # reads  data/{corpus}/ref.json and hyps/{system}/{corpus}.json
    # writes results/{corpus}/{system}/pairs.json, baselines.json, {scorer}.json

Use --ref/--hyp to point at other files, e.g. the bundled example:

    python scripts/run_all.py --corpus example --system example \
        --ref examples/ref.json --hyp examples/hyp.json
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from semscore.aggregate import run_scorer  # noqa: E402
from semscore.pairs import extract_pairs, load_pairs, save_pairs  # noqa: E402
from semscore.scorers import PRIORITY, REGISTRY, get_scorer  # noqa: E402


def set_seeds(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)


def baselines(ref, hyp, collar, normalizer, regex, uem):
    import meeteval
    from tcpsemer.core import _load
    # Load (and normalise) once with float timestamps: meeteval's tcpwer
    # fails on Decimal timestamps with a float collar.
    ref, hyp = _load(ref, hyp, regex, uem, normalizer)
    out = {}
    for name, fn, extra in [("cpWER", meeteval.wer.cpwer, {}),
                            ("tcpWER", meeteval.wer.tcpwer, {"collar": collar})]:
        per = fn(ref, hyp, **extra)
        c = meeteval.wer.combine_error_rates(per)
        out[name] = {"error_rate": float(c.error_rate), "errors": int(c.errors),
                     "length": int(c.length),
                     "sessions": {k: float(v.error_rate) for k, v in per.items()}}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--ref", help="default: data/{corpus}/ref.json")
    p.add_argument("--hyp", help="default: hyps/{system}/{corpus}.json")
    p.add_argument("--out", default="results")
    p.add_argument("--scorers", nargs="+", default=None,
                   help=f"default: priority set {PRIORITY}; 'all' for every "
                        f"scorer. Choices: {sorted(REGISTRY)}")
    p.add_argument("--collar", type=float, default=5.0)
    p.add_argument("--max-gap", type=float, default=2.0)
    p.add_argument("--normalizer", default=None, help="e.g. chime8")
    p.add_argument("--regex", default=None)
    p.add_argument("--uem", default=None)
    p.add_argument("--context-turns", type=int, default=3)
    p.add_argument("--tag", default="",
                   help="suffix for the output folder, e.g. '_chime7' for a "
                        "second normalizer (Experiment 3)")
    p.add_argument("--realign", action="store_true",
                   help="recompute pairs.json even if it exists")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    set_seeds(a.seed)
    ref = a.ref or os.path.join("data", a.corpus, "ref.json")
    hyp = a.hyp or os.path.join("hyps", a.system, f"{a.corpus}.json")
    out = os.path.join(a.out, a.corpus, a.system + a.tag)
    os.makedirs(out, exist_ok=True)

    names = a.scorers or PRIORITY
    if names == ["all"]:
        names = list(REGISTRY)

    pairs_path = os.path.join(out, "pairs.json")
    if os.path.exists(pairs_path) and not a.realign:
        pairs = load_pairs(pairs_path)
        print(f"using existing {pairs_path}")
    else:
        t = time.time()
        pairs = extract_pairs(ref, hyp, collar=a.collar, regex=a.regex,
                              uem=a.uem, normalizer=a.normalizer,
                              max_gap=a.max_gap, context_turns=a.context_turns)
        save_pairs(pairs, pairs_path)
        b = baselines(ref, hyp, a.collar, a.normalizer, a.regex, a.uem)
        with open(os.path.join(out, "baselines.json"), "w") as f:
            json.dump(b, f, indent=1)
        print(f"aligned {len(pairs['sessions'])} sessions, "
              f"{len(pairs['records'])} records in {time.time() - t:.1f}s; "
              f"cpWER={b['cpWER']['error_rate']:.2%} "
              f"tcpWER={b['tcpWER']['error_rate']:.2%}")

    for name in names:
        t = time.time()
        res = run_scorer(pairs, get_scorer(name))
        res["pairs"] = pairs_path
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump(res, f, indent=1)
        c = res["corpus"]
        print(f"{name:16s} rate={c['error_rate']:.2%}  mean_d={c['mean_d']:.3f}"
              f"  ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
