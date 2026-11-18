"""Tests for the pluggable scorers (README_TODO.md §1-§2, §4).

The embedding tests download small models on first run
(all-MiniLM-L6-v2, distilbert-base-uncased for BERTScore).
"""
import random

import pytest

from semscore.aggregate import aggregate, run_scorer, score_records
from semscore.pairs import extract_pairs
from semscore.perturb import (contraction, entity, filler, make_perturbations,
                              negation, number, repetition)
from semscore.scorers import (BERTScoreScorer, BLEUScorer, ChrFScorer,
                              ContextScorer, NLIScorer, SemDistScorer, WERScorer,
                              word_edit_distance)

REF, HYP = "examples/ref.json", "examples/hyp.json"

SENT = "i do not want to move the meeting to friday"
FILLER_DEL = ("so i uh think we should move the meeting to friday",
              "so i think we should move the meeting to friday")
NOT_DROP = (SENT, "i do want to move the meeting to friday")


@pytest.fixture(scope="module")
def minilm():
    return SemDistScorer()


@pytest.fixture(scope="module")
def ctx0():
    return ContextScorer(k=0)


@pytest.fixture(scope="module")
def ctx3():
    return ContextScorer(k=3)


@pytest.fixture(scope="module")
def bertscore():
    return BERTScoreScorer("distilbert-base-uncased")


@pytest.fixture(scope="module")
def nli():
    return NLIScorer()


@pytest.fixture(scope="module")
def pairs():
    return extract_pairs(REF, HYP)


def _all(minilm, ctx0, ctx3, bertscore):
    return [WERScorer(), ChrFScorer(), BLEUScorer(), minilm, ctx0, ctx3, bertscore]


# ---- hand-made pairs -------------------------------------------------------

def test_identical_is_zero(minilm, ctx0, ctx3, bertscore):
    for s in _all(minilm, ctx0, ctx3, bertscore):
        assert s.dissim(SENT, SENT, ["hello there", "how are you"]) == \
            pytest.approx(0.0, abs=1e-4), s.name


def test_empty_hypothesis_is_one(pairs):
    p = {"sessions": {"S": {"length": 4, "unassigned_hyp_words": 0}},
         "records": [dict(session_id="S", kind="utt", speaker="A",
                          start_time=0, end_time=1, ref_text="a b c d",
                          hyp_text="", n_ref=4, n_hyp=0, overlap=False,
                          overlap_frac=0, context=[])]}
    d = score_records(p["records"], WERScorer())
    assert d == [1.0]
    assert aggregate(p, d)["corpus"]["error_rate"] == 1.0


def test_insertion_and_missed_rules():
    p = {"sessions": {"S": {"length": 3, "unassigned_hyp_words": 2}},
         "records": [
             dict(session_id="S", kind="insertion", speaker="A", start_time=0,
                  end_time=1, ref_text="", hyp_text="x y z", n_ref=0, n_hyp=3,
                  overlap=False, overlap_frac=0, context=[]),
             dict(session_id="S", kind="missed", speaker="B", start_time=None,
                  end_time=None, ref_text="a b c", hyp_text="", n_ref=3,
                  n_hyp=0, overlap=None, overlap_frac=None, context=[])]}
    d = score_records(p["records"], WERScorer())
    assert aggregate(p, d)["corpus"]["errors"] == 3 + 3 + 2


@pytest.mark.parametrize("name", ["minilm", "ctx0", "ctx3", "bertscore", "nli"])
def test_filler_deletion_small_negation_large(name, request):
    s = request.getfixturevalue(name)
    d_filler = s.dissim(*FILLER_DEL, ["we need to talk about the schedule"])
    d_not = s.dissim(*NOT_DROP, ["we need to talk about the schedule"])
    assert d_filler < d_not, (name, d_filler, d_not)
    assert d_filler < 0.1, (name, d_filler)


def test_nli_bidirectional(nli):
    assert nli.dissim(SENT, SENT) < 0.1
    # dropping content keeps R => H but breaks H => R
    assert nli.dissim("we met on monday and fixed the bug", "we met on monday") > 0.5


def test_wer_cannot_tell_them_apart():
    # one edited word each: that is why the perturbation classes are matched
    assert WERScorer().dissim(*FILLER_DEL) == pytest.approx(1 / 11)
    assert WERScorer().dissim(*NOT_DROP) == pytest.approx(1 / 10)


def test_wer_clip():
    assert WERScorer().dissim("a", "b c d") == 1.0
    assert WERScorer(clip=False).dissim("a", "b c d") == 3.0


def test_semdist_clip():
    s = SemDistScorer()
    assert all(0 <= x <= 1 for x in s.dissim_batch(["yes"], ["no thanks"]))


def test_context_k0_close_to_semdist(minilm, ctx0):
    a = minilm.dissim(*NOT_DROP)
    b = ctx0.dissim(*NOT_DROP)
    assert abs(a - b) < 0.05


def test_context_is_shared_and_cached(ctx3):
    ctx = ["are you coming tomorrow", "i am not sure"]
    a = ctx3.dissim("yeah", "yes", ctx)
    n = len(ctx3.cache)
    assert ctx3.dissim("yeah", "yes", ctx) == a
    assert len(ctx3.cache) == n


# ---- reproduction checks ---------------------------------------------------

def test_reproduces_tcpsemer(pairs):
    from tcpsemer import combine, tcpsemer
    orig = combine(tcpsemer(REF, HYP)).error_rate
    ours = run_scorer(pairs, SemDistScorer(clip=False))["corpus"]["error_rate"]
    assert ours == pytest.approx(orig, abs=1e-6)


def test_wer_scorer_matches_tcpwer(pairs):
    import meeteval
    tcpwer = meeteval.wer.combine_error_rates(
        meeteval.wer.tcpwer(REF, HYP, collar=5)).error_rate
    ours = run_scorer(pairs, WERScorer(clip=False))["corpus"]["error_rate"]
    assert ours == pytest.approx(float(tcpwer), abs=0.02)


def test_pairs_fields(pairs):
    r = pairs["records"][1]
    assert r["kind"] == "utt"
    assert r["context"] == ["hello how are you doing today", "i am fine thank you"]
    assert pairs["sessions"]["S1"] == {"length": 17, "unassigned_hyp_words": 0}


# ---- perturbations ---------------------------------------------------------

def test_edits():
    rng = random.Random(0)
    w = "i do not think it is ready for friday at two".split()
    for seed in range(5):
        neg = negation(w, random.Random(seed))
        assert word_edit_distance(neg, w) == 1
        assert neg.count("not") != w.count("not") or "isn't" in neg
    assert number(w, rng) != w and word_edit_distance(number(w, rng), w) == 1
    assert entity(w, rng) != w and word_edit_distance(entity(w, rng), w) == 1
    assert contraction(w, rng) in (
        "i don't think it is ready for friday at two".split(),
        "i do not think it's ready for friday at two".split())
    assert len(repetition(w, rng)) > len(w)
    assert filler(w, rng) != w


def test_balanced_set():
    utts = [{"ref_text": t} for t in [
        "i do not think it is ready for friday",
        "we should start the meeting at two",
        "yes i will send it before monday",
        "there is no reason to move it to tuesday",
        "is this the first time we have met"]]
    rows = make_perturbations(utts, n=10, seed=0)
    assert rows
    from collections import Counter
    c = Counter((r["n_edit"], r["label"]) for r in rows)
    for e in {k[0] for k in c}:
        assert c[(e, 0)] == c[(e, 1)]
