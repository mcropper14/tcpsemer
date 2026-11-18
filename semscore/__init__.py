"""Pluggable segment scorers on top of the tcpSemER alignment.

Course project: Characterizing Semantic Significance of Word Recognition
Errors. See README_TODO.md and semscore/README.md.
"""
from .aggregate import aggregate, run_scorer, score_records
from .pairs import extract_pairs, load_pairs, save_pairs
from .scorers import (BERTScoreScorer, BLEUScorer, ChrFScorer, ContextScorer,
                      REGISTRY, Scorer, SemDistScorer, WERScorer, get_scorer)

__all__ = ["extract_pairs", "load_pairs", "save_pairs", "run_scorer",
           "score_records", "aggregate", "Scorer", "WERScorer", "ChrFScorer",
           "BLEUScorer", "SemDistScorer", "BERTScoreScorer", "ContextScorer",
           "REGISTRY", "get_scorer"]
