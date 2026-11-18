"""Segment-level scoring functions for aligned (reference, hypothesis) pairs.

Every scorer returns a dissimilarity ``d``; the aggregation in
:mod:`semscore.aggregate` turns it into errors as ``|R| * d``. Scorers are only
called on pairs where both sides are non-empty; empty sides are handled by
the aggregation rules (deletion: d = 1, insertion: |H| errors).

``d`` is in [0, 1] for every scorer here, except ``SemDistScorer(clip=False)``
and ``ContextScorer(clip=False)``, which keep the raw ``1 - cos`` in [0, 2]
exactly like the original tcpSemER code.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

Context = Optional[List[str]]


def _key(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def _device(device):
    if device is not None:
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Scorer:
    """Base class. Subclasses implement ``dissim`` or ``dissim_batch``."""

    name: str = "scorer"
    uses_context: bool = False

    def dissim(self, ref: str, hyp: str, ctx: Context = None) -> float:
        return self.dissim_batch([ref], [hyp], [ctx])[0]

    def dissim_batch(self, refs: Sequence[str], hyps: Sequence[str],
                     ctxs: Optional[Sequence[Context]] = None) -> List[float]:
        ctxs = ctxs if ctxs is not None else [None] * len(refs)
        return [self.dissim(r, h, c) for r, h, c in zip(refs, hyps, ctxs)]

    def config(self) -> Dict:
        return {"name": self.name}


# --------------------------------------------------------------------------
# Edit distance and n-gram scorers
# --------------------------------------------------------------------------

def word_edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


class WERScorer(Scorer):
    """d = min(1, edit_distance / |R|). With ``clip=False``, d * |R| is the
    plain word edit distance of the utterance (used for the tcpWER check)."""

    def __init__(self, clip: bool = True):
        self.clip = clip
        self.name = "wer" if clip else "wer-noclip"

    def dissim(self, ref, hyp, ctx=None):
        r, h = ref.split(), hyp.split()
        d = word_edit_distance(r, h) / max(len(r), 1)
        return min(1.0, d) if self.clip else d

    def config(self):
        return {"name": self.name, "clip": self.clip}


class ChrFScorer(Scorer):
    """d = 1 - chrF / 100 (sacrebleu ``sentence_chrf``, default chrF2)."""

    def __init__(self, word_order: int = 0):
        self.word_order = word_order
        self.name = "chrf" if word_order == 0 else f"chrf{'+' * word_order}"

    def dissim(self, ref, hyp, ctx=None):
        from sacrebleu.metrics import CHRF
        if not hasattr(self, "_m"):
            self._m = CHRF(word_order=self.word_order)
        return 1.0 - self._m.sentence_score(hyp, [ref]).score / 100.0

    def config(self):
        return {"name": self.name, "word_order": self.word_order}


class BLEUScorer(Scorer):
    """d = 1 - sentence BLEU / 100 with exponential smoothing
    (sacrebleu ``sentence_bleu`` defaults)."""

    name = "bleu"

    def __init__(self, smooth_method: str = "exp"):
        self.smooth_method = smooth_method

    def dissim(self, ref, hyp, ctx=None):
        from sacrebleu.metrics import BLEU
        if not hasattr(self, "_m"):
            self._m = BLEU(effective_order=True, smooth_method=self.smooth_method)
        return 1.0 - self._m.sentence_score(hyp, [ref]).score / 100.0

    def config(self):
        return {"name": self.name, "smooth_method": self.smooth_method}


# --------------------------------------------------------------------------
# Embedding scorers
# --------------------------------------------------------------------------

class SemDistScorer(Scorer):
    """SemDist: d = 1 - max(0, cos(E(R), E(H))) with sentence-transformers.

    ``clip=False`` gives ``1 - cos`` (the original tcpSemER distance).
    Embeddings are cached by a hash of the text.
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 name: Optional[str] = None, clip: bool = True,
                 device: Optional[str] = None, batch_size: int = 64):
        self.model_name = model if isinstance(model, str) else type(model).__name__
        self._model = None if isinstance(model, str) else model
        self.name = name or f"semdist[{self.model_name.split('/')[-1]}]"
        self.clip, self.device, self.batch_size = clip, device, batch_size
        self.cache: Dict[str, "object"] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name,
                                              device=_device(self.device))
        return self._model

    def embed(self, texts: Sequence[str]):
        import torch
        missing = list(dict.fromkeys(t for t in texts if _key(t) not in self.cache))
        if missing:
            emb = self.model.encode(missing, batch_size=self.batch_size,
                                    convert_to_tensor=True,
                                    normalize_embeddings=True,
                                    show_progress_bar=False).float().cpu()
            for t, e in zip(missing, emb):
                self.cache[_key(t)] = e
        return torch.stack([self.cache[_key(t)] for t in texts])

    def dissim_batch(self, refs, hyps, ctxs=None):
        if not refs:
            return []
        a, b = self.embed(list(refs)), self.embed(list(hyps))
        cos = (a * b).sum(-1).tolist()
        return [1.0 - (max(0.0, c) if self.clip else c) for c in cos]

    def config(self):
        return {"name": self.name, "model": self.model_name, "clip": self.clip}


class BERTScoreScorer(Scorer):
    """d = clip(1 - BERTScore F1, 0, 1), with baseline rescaling by default.

    Rescaled F1 can fall slightly outside [0, 1], hence the clipping.
    """

    def __init__(self, model_type: str = "roberta-large", rescale: bool = True,
                 lang: str = "en", name: Optional[str] = None,
                 device: Optional[str] = None, batch_size: int = 64):
        self.model_type, self.rescale, self.lang = model_type, rescale, lang
        self.device, self.batch_size = device, batch_size
        self.name = name or f"bertscore[{model_type.split('/')[-1]}]"
        self._scorer = None
        self.cache: Dict[str, float] = {}

    def dissim_batch(self, refs, hyps, ctxs=None):
        if not refs:
            return []
        keys = [_key(r, h) for r, h in zip(refs, hyps)]
        todo = {k: (r, h) for k, r, h in zip(keys, refs, hyps) if k not in self.cache}
        if todo:
            if self._scorer is None:
                from bert_score import BERTScorer
                self._scorer = BERTScorer(
                    model_type=self.model_type, lang=self.lang,
                    rescale_with_baseline=self.rescale,
                    device=_device(self.device), batch_size=self.batch_size)
            ks = list(todo)
            _, _, f1 = self._scorer.score([todo[k][1] for k in ks],
                                          [todo[k][0] for k in ks])
            for k, f in zip(ks, f1.tolist()):
                self.cache[k] = min(1.0, max(0.0, 1.0 - f))
        return [self.cache[k] for k in keys]

    def config(self):
        return {"name": self.name, "model_type": self.model_type,
                "rescale_with_baseline": self.rescale, "lang": self.lang}


class ContextScorer(Scorer):
    """Context-aware embedding distance.

    The input is ``C [SEP] utterance`` where ``C`` is the previous ``k``
    reference turns (same ``C`` for R and H). The token-level encoder output
    is mean-pooled over the target utterance's tokens only (special tokens
    and context tokens are excluded). ``k = 0`` encodes the utterance alone
    with the same pooling, so it is close to (not identical to) SemDist with
    the same model, which also pools [CLS]/[SEP].

    If the context does not fit in ``max_length`` tokens it is truncated from
    the left (oldest words dropped first).
    """

    uses_context = True

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 k: int = 3, clip: bool = True, name: Optional[str] = None,
                 device: Optional[str] = None, batch_size: int = 32,
                 max_length: int = 256):
        self.model_name, self.k, self.clip = model, k, clip
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.name = name or f"ctx[{model.split('/')[-1]},k={k}]"
        self._tok = self._enc = None
        self.cache: Dict[str, "object"] = {}

    def _load(self):
        if self._enc is None:
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._tok.truncation_side = "left"
            self._enc = AutoModel.from_pretrained(self.model_name).to(
                _device(self.device)).eval()

    def _ctx_text(self, ctx: Context) -> str:
        if self.k <= 0 or not ctx:
            return ""
        return " ".join(ctx[-self.k:])

    def _encode(self, items):
        """items: list of (context_text, utterance). Returns pooled vectors."""
        import torch
        self._load()
        dev = self._enc.device
        out = []
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            with_ctx = [j for j, (c, _) in enumerate(chunk) if c]
            vecs = [None] * len(chunk)
            groups = [(with_ctx, True),
                      ([j for j in range(len(chunk)) if j not in with_ctx], False)]
            for idx, use_ctx in groups:
                if not idx:
                    continue
                if use_ctx:
                    enc = self._tok([chunk[j][0] for j in idx],
                                    [chunk[j][1] for j in idx],
                                    truncation="only_first",
                                    max_length=self.max_length,
                                    padding=True, return_tensors="pt")
                    target = 1
                else:
                    enc = self._tok([chunk[j][1] for j in idx], truncation=True,
                                    max_length=self.max_length, padding=True,
                                    return_tensors="pt")
                    target = 0
                mask = torch.tensor(
                    [[sid == target for sid in enc.sequence_ids(b)]
                     for b in range(len(idx))], dtype=torch.float32)
                with torch.no_grad():
                    h = self._enc(**{k: v.to(dev) for k, v in enc.items()}
                                  ).last_hidden_state.float().cpu()
                pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                for j, v in zip(idx, pooled):
                    vecs[j] = v
            out.extend(vecs)
        return out

    def _embed(self, items):
        import torch
        keys = [_key(c, u) for c, u in items]
        todo = {}
        for k, it in zip(keys, items):
            if k not in self.cache:
                todo[k] = it
        if todo:
            for k, v in zip(todo, self._encode(list(todo.values()))):
                self.cache[k] = v
        return torch.stack([self.cache[k] for k in keys])

    def _fits(self, ctx_text: str, utt: str) -> str:
        # only_first truncation fails if the utterance alone is too long;
        # drop the context in that case.
        if ctx_text and len(self._tok(utt)["input_ids"]) >= self.max_length - 2:
            return ""
        return ctx_text

    def dissim_batch(self, refs, hyps, ctxs=None):
        if not refs:
            return []
        self._load()
        ctxs = ctxs if ctxs is not None else [None] * len(refs)
        cts = [self._ctx_text(c) for c in ctxs]
        a = self._embed([(self._fits(c, r), r) for c, r in zip(cts, refs)])
        b = self._embed([(self._fits(c, h), h) for c, h in zip(cts, hyps)])
        cos = (a * b).sum(-1).tolist()
        return [1.0 - (max(0.0, c) if self.clip else c) for c in cos]

    def config(self):
        return {"name": self.name, "model": self.model_name, "k": self.k,
                "clip": self.clip, "max_length": self.max_length,
                "context": "previous k reference turns, all speakers"}


# --------------------------------------------------------------------------
# NLI scorer (bidirectional entailment)
# --------------------------------------------------------------------------

class NLIScorer(Scorer):
    """d = 1 - min(P(R => H), P(H => R)) from an MNLI cross-encoder.

    Requiring entailment in both directions penalises dropped content (breaks
    H => R) as well as added content (breaks R => H). Probabilities are
    cached per ordered (premise, hypothesis) pair.
    """

    def __init__(self, model: str = "microsoft/deberta-large-mnli",
                 name: Optional[str] = None, device=None, batch_size: int = 32,
                 max_length: int = 256):
        self.model_name, self.device = model, device
        self.batch_size, self.max_length = batch_size, max_length
        self.name = name or f"nli[{model.split('/')[-1]}]"
        self._tok = self._clf = None
        self._ent = None
        self.cache: Dict[str, float] = {}

    def _load(self):
        if self._clf is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._clf = AutoModelForSequenceClassification.from_pretrained(
                self.model_name).to(_device(self.device)).eval()
            labels = {v.lower(): k for k, v in self._clf.config.id2label.items()}
            self._ent = next(i for l, i in labels.items() if l.startswith("entail"))

    def _p_entail(self, premises, hypotheses) -> List[float]:
        import torch
        self._load()
        keys = [_key("nli", p, h) for p, h in zip(premises, hypotheses)]
        todo = {k: (p, h) for k, p, h in zip(keys, premises, hypotheses)
                if k not in self.cache}
        items = list(todo.items())
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            enc = self._tok([p for _, (p, _) in chunk], [h for _, (_, h) in chunk],
                            truncation=True, max_length=self.max_length,
                            padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self._clf(**{k: v.to(self._clf.device)
                                      for k, v in enc.items()}).logits.float()
            probs = logits.softmax(-1)[:, self._ent].cpu().tolist()
            for (k, _), pr in zip(chunk, probs):
                self.cache[k] = pr
        return [self.cache[k] for k in keys]

    def dissim_batch(self, refs, hyps, ctxs=None):
        if not refs:
            return []
        fwd = self._p_entail(refs, hyps)
        bwd = self._p_entail(hyps, refs)
        return [1.0 - min(a, b) for a, b in zip(fwd, bwd)]

    def config(self):
        return {"name": self.name, "model": self.model_name,
                "d": "1 - min(P(R=>H), P(H=>R))", "max_length": self.max_length}


# --------------------------------------------------------------------------
# Registry used by the scripts
# --------------------------------------------------------------------------

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MPNET = "sentence-transformers/all-mpnet-base-v2"
# Placeholder choice for "one larger general-purpose embedding model";
# the team should confirm it.
LARGE = "BAAI/bge-large-en-v1.5"

REGISTRY = {
    "wer": lambda: WERScorer(),
    "wer-noclip": lambda: WERScorer(clip=False),
    "chrf": lambda: ChrFScorer(),
    "bleu": lambda: BLEUScorer(),
    "semdist-minilm": lambda: SemDistScorer(MINILM, name="semdist-minilm"),
    "semdist-mpnet": lambda: SemDistScorer(MPNET, name="semdist-mpnet"),
    "semdist-large": lambda: SemDistScorer(LARGE, name="semdist-large"),
    # The original tcpSemER distance (1 - cos, unclipped, MiniLM).
    "tcpsemer-orig": lambda: SemDistScorer(MINILM, name="tcpsemer-orig", clip=False),
    "bertscore": lambda: BERTScoreScorer("roberta-large", name="bertscore"),
    "ctx-k0": lambda: ContextScorer(MINILM, k=0, name="ctx-k0"),
    "ctx-k1": lambda: ContextScorer(MINILM, k=1, name="ctx-k1"),
    "ctx-k3": lambda: ContextScorer(MINILM, k=3, name="ctx-k3"),
    "nli": lambda: NLIScorer("microsoft/deberta-large-mnli", name="nli"),
}

# The TODO's priority set: WER, chrF, one SemDist, context k=0/3.
PRIORITY = ["wer", "chrf", "semdist-minilm", "ctx-k0", "ctx-k3"]


def get_scorer(name: str) -> Scorer:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown scorer {name!r}; choose from {sorted(REGISTRY)}")
