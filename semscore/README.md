# semscore: pluggable scorers on the tcpSemER alignment

The tcpSemER speaker assignment and time alignment stay fixed. Only the
function that scores each aligned (reference, hypothesis) utterance changes.
`tcpsemer/` (the upstream package) is not modified.

## Setup

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
python -m pytest -q          # downloads MiniLM and distilbert on first run
```

## Layout

```
data/{corpus}/ref.json                  SegLST references (git-ignored)
hyps/{system}/{corpus}.json             SegLST system outputs (git-ignored)
results/{corpus}/{system}/pairs.json    aligned pairs, computed once
results/{corpus}/{system}/baselines.json  cpWER, tcpWER (MeetEval)
results/{corpus}/{system}/{scorer}.json   one file per scorer
```

## Pipeline

```bash
# §3: align once, then run scorers (default: wer chrf semdist-minilm ctx-k0 ctx-k3)
python scripts/run_all.py --corpus notsofar1_dev --system pyannote_whisper --scorers all
python scripts/make_table.py results --latex

# §4: perturbations
python scripts/perturbations.py make --pairs results/*/*/pairs.json --out results/perturb/perturbed.jsonl
python scripts/perturbations.py score --set results/perturb/perturbed.jsonl --out results/perturb --scorers all

# §5: agreement between scorers (correlations, heatmap, top disagreements)
python scripts/agreement.py results/notsofar1_dev/pyannote_whisper --a wer --b semdist-minilm

# §6: a second normaliser goes in a sibling folder
python scripts/run_all.py --corpus notsofar1_dev --system pyannote_whisper --normalizer chime8 --tag _chime8
```

Try it on the bundled example with
`--corpus example --system example --ref examples/ref.json --hyp examples/hyp.json`.

## Definitions (what the code does)

Per aligned record (see `semscore/pairs.py`):

| record | errors |
|---|---|
| utterance, hypothesis non-empty | \|R\| · d(R, H) |
| utterance, hypothesis empty (deletion) | \|R\| (d = 1) |
| inserted words outside every utterance ± collar | \|H\| |
| missed reference speaker | \|R\| |
| words of an unassigned hypothesis speaker | their count |

Rate = total errors / N_ref (micro average). `mean_d` is the unweighted mean
of d over utterances, with deletions counted as d = 1.

Scorers (`semscore/scorers.py`, names in `REGISTRY`):

| name | d |
|---|---|
| `wer` | min(1, edit distance / \|R\|) |
| `wer-noclip` | edit distance / \|R\|. Gives the same corpus rate as tcpWER on the example |
| `chrf`, `bleu` | 1 − score/100 (sacrebleu; BLEU uses exp smoothing and effective order) |
| `semdist-minilm/-mpnet/-large` | 1 − max(0, cos), normalised sentence-transformers embeddings |
| `tcpsemer-orig` | 1 − cos, unclipped, MiniLM. Gives the same score as the original tcpSemER |
| `bertscore` | clip(1 − F1, 0, 1), roberta-large, **baseline rescaling on** |
| `ctx-k{0,1,3}` | 1 − max(0, cos) of MiniLM token states mean-pooled over the target utterance only, encoded as `C [SEP] U`, C = previous k reference turns (all speakers) |

`semdist-large` is set to `BAAI/bge-large-en-v1.5` as a placeholder. The team
still needs to choose this model.

A new scorer subclasses `Scorer`, implements `dissim(ref, hyp, ctx)` (or
`dissim_batch`) and is added to `REGISTRY`.
