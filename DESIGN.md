# biascompare — Design (POC)

> **POC design record.** This is the design document of the POC this framework was forked
> from. It still describes the code in `src/` accurately. The framework plan is in
> [`docs/PLAN.md`](./docs/PLAN.md) and the data specification in [`docs/DATA.md`](./docs/DATA.md).

A proof of concept for a **comparative model-bias testing framework**. Given a set of
Hugging Face models, a dataset and an optional instruction prompt, it measures each
model's **gender bias beyond its general sensitivity to wording** and compares every
model against a reference model (model 0). Output: one JSON file.

The POC only has to show one thing: **given 2 models, we can compare their biases.**

This document records every design decision made during planning, and why, so that an
agent building the full version has the context the code alone does not carry.

---

## 1. Grounding: the paper

Yang, Levy, Goldberg & Wallace (2026), *"Compared to What? Baselines and Metrics for
Counterfactual Prompting"*, arXiv:2605.01048.

- **Core claim.** If a demographic swap (he→she) changes a model's output, that alone
  is not evidence of bias: models react to *any* textual change. Every intervention
  expressed in text bundles the attribute with its wording ("treatment variation
  irrelevance" is violated).
- **Fix.** Compare the targeted effect against a **baseline perturbation of matched
  magnitude** that should be irrelevant (paraphrase, token-change % matched to within
  0.5 pp). Attribute an effect to the attribute only if it exceeds the baseline.
- **Test.** For per-sample divergence `d` (JSD/KL):
  `H0: E[Δi] ≤ 0`, where `Δi = d(p_i, p_i') − d(p_i, p_i'')`
  (`p'` = targeted, `p''` = baseline); one-sided paired t-test across samples,
  Bonferroni within hypothesis groups. The paper used **one** baseline per sample.
- **Metric advice.** Per-sample distributional metrics (JSD, KL) are far more powerful
  than aggregate ones (flip rate, MI, φ), especially under class imbalance. Use a
  signed measure (regression coefficient) for directional hypotheses.

## 2. Scope of the POC

| Decision | Choice | Why |
|---|---|---|
| Task / domain | Toxicity classification of social-media posts (Twitter, Reddit) | Target use case: content moderation |
| Model type | Fine-tuned HF **text-classification** models (BERT family) | Gives a clean class distribution; no prompt engineering |
| Models (POC) | model 0 `unitary/toxic-bert`, model 1 `s-nlp/roberta_toxicity_classifier` | Two unrelated classifiers — the POC shows "any 2 models can be compared". Also exercises both label formats (multi-label sigmoid vs 2-class softmax) |
| Protected axis | **Gender only** | Same as the paper; more axes later |
| Dataset | **10 posts**, hand-written, synthetic | Enough to exercise the stats; data generation deliberately kept simple |
| Control perturbations | **K = 5** synonym swaps per post (configurable) | Paper used 1 and names more as future work; averaging K reduces noise in Δi; K=1 reproduces the paper exactly |
| Output | **JSON only** | Rudimentary is fine; readable views can be generated later from the JSON |
| Interface | A notebook that **only calls** `src/` | The logic lives in the package so it can be reused and tested |
| Runtime | **Google Colab**, pulling from a public GitHub repo | See §9 |

Explicitly **out of scope** for the POC: the larger "fine-tuning as bias mitigation"
project, other axes, masked-LM or generative scorers, paraphrase-based controls,
markdown/HTML reports.

## 3. Inputs

```python
ModelSpec(name: str, toxic_label: str = "toxic", token: str | None = None)
compare_models(models, dataset, instruction=None, n_controls=5, seed=0, alpha=0.05, ...)
```

- **Models** — HF repo id plus the name of the label that means "toxic" in that model
  (label names differ between models: `toxic`, `toxicity`, `offensive`, …).
- **Access key** — optional `token` per model. If absent, fall back to the `HF_TOKEN`
  environment variable, then to the cached `huggingface-cli login` (i.e. pass
  `token=None` to `from_pretrained`). **Tokens are never written to results or
  logs** (`repr=False`, excluded from serialisation).
- **Dataset** — JSONL, one post per line:
  ```json
  {"id": "p03", "platform": "reddit", "text": "...", "intended_toxicity": "borderline", "gender": "female"}
  ```
  `gender` is the gender the post originally references (`male` | `female`);
  `intended_toxicity` ∈ {clean, mild, borderline, rude, toxic} is an authoring note only.
- **Instruction prompt (optional)** — kept in the interface because future scorers
  (masked-LM, generative LLM) will use it. **Classifiers ignore it** (prepending it to
  the post would corrupt the classifier's input distribution). Results record
  `instruction_used: false` for each classifier.

## 4. Scoring

`Scorer` protocol → for a batch of texts return `[P(non-toxic), P(toxic)]` per text.

`HFClassifierScorer` (the only implementation in the POC):
- Loads `AutoTokenizer` / `AutoModelForSequenceClassification` with the token.
- Resolves `toxic_label` against `config.id2label` (case-insensitive); raises a clear
  `ValueError` listing the available labels if it is missing.
- If `config.problem_type == "multi_label_classification"`: `p = sigmoid(logit[toxic])`;
  other heads (severe_toxic, insult, …) are ignored. Otherwise: softmax over all
  labels, `p = probs[toxic]`. Returns `[1 − p, p]`.
- Device: `cuda` if available, else `mps`, else `cpu`. Default float32, no dtype
  kwargs (keeps it portable across transformers 4.x/5.x on Colab). Batched,
  `torch.inference_mode()`, truncation to the model max length.
- One model is loaded at a time and freed before the next.

Extension point: new scorer types (masked-LM, LLM next-token over options) implement
the same protocol and set `uses_instruction = True`.

## 5. Perturbations

Both perturbations are generated **once per post and shared across all models**, so
every model sees identical inputs. Seeded.

### 5.1 Targeted: gender swap (`perturb/gender.py`)
- Bidirectional lexicon swap (`resources/gender_lexicon.json`): pronouns
  (he/she, him/her, his/her, himself/herself), gendered nouns (man/woman, guy/girl,
  boyfriend/girlfriend, …), kinship terms, and common first names.
- Case preserved (`He`→`She`, `HE`→`SHE`).
- **"her" disambiguation** (her → him | his): `him` if the next token is
  punctuation/end-of-text or one of a small set of function words
  (to, and, the, a, that, with, for, …); otherwise `his`.
- Social-media traps are **avoided in the data, not handled in code** (documented for
  the real build): interjection "Man, …", generic "you guys", slang address terms
  (bro/dude/sis) with no clean counterpart.

### 5.2 Controls: matched-magnitude synonym swaps (`perturb/controls.py`)
- Replace **n** words with synonyms, where **n = number of words the gender swap
  changed** for that post (word-count matching; see deviations).
- Candidate words: in `resources/synonyms.json`, and **not** protected. Protected =
  gender lexicon ∪ `resources/protected_words.json` (toxic/insult words — swapping
  "stupid"→"smart" would change the label and inflate the baseline, hiding bias).
- Generate **K distinct** variants (default 5), up to a bounded number of attempts.
  Duplicates are discarded. Record `effective_k`; flag posts with `effective_k < K`
  and posts where fewer than n candidate words exist (`n_matched < n`).
- A post with **zero** usable candidate words (`effective_k == 0`, no baseline to
  compare against at all) is flagged `no_controls` and excluded from statistics like
  `gender_mismatch`/`no_gender_terms` (§6), though it is still scored and kept in
  `posts`.
- Hand-written synonym list is a POC simplification. Planned upgrade: masked-LM
  substitution with a separate helper model (e.g. `distilroberta-base`) + semantic
  similarity filter (e.g. `all-MiniLM-L6-v2`). LLM paraphrase (the paper's method) is
  a poor fit for toxic text: chat models sanitise or refuse, changing the label.

## 6. Metrics and statistics

Per post *i* and model *m* (`p` = `[P(non-toxic), P(toxic)]`):

| Quantity | Definition |
|---|---|
| `jsd_gender` | `JSD(p_orig, p_gender)` (base 2, in [0, 1]) |
| `jsd_controls` | `JSD(p_orig, p_ctrl_k)` for k = 1..K |
| `delta` (Δi) | `jsd_gender − mean_k(jsd_controls)` |
| `direction` | `P_toxic(female-referencing version) − P_toxic(male-referencing version)`. If the post is female-original: `p_orig − p_swap`; male-original: `p_swap − p_orig` |
| `flip_gender` | predicted label (P(toxic) ≥ 0.5) differs between original and gender swap |
| `flip_controls_rate` | share of controls whose predicted label differs from the original |

Balancing the dataset 5 male / 5 female original posts matters: otherwise "female
version" would be confounded with "perturbed version" and `direction` would be
meaningless.

A post flagged `gender_mismatch`, `no_gender_terms`, or `no_controls` (§5.1, §5.2) is
excluded from every statistic below (its `delta`/`direction` never enter a mean, t-test,
or bootstrap for any model) but still appears in `posts` and each model's `per_post`, so
one bad post can never silently NaN-poison the rest of a model's summary.

**Per model** (is there bias beyond noise?):
- `sensitivity` = mean of all control JSDs (the model's prompt sensitivity)
- `raw_gender_effect` = mean `jsd_gender`
- `mean_delta`, one-sided paired t-test `H1: E[Δ] > 0` (the paper's test), Bonferroni
  across models → `p_delta`, `p_delta_bonferroni`
- `mean_direction`, bootstrap 95% CI, two-sided one-sample t-test → `p_direction`
- flip rates

**Model k vs model 0** (the comparative question):
- `delta_diff_i = Δi(k) − Δi(0)`: mean, bootstrap 95% CI, two-sided Wilcoxon
  signed-rank p
- `direction_diff_i = direction_i(k) − direction_i(0)`: same
- Bootstrap: 2000 resamples, seeded. Degenerate inputs (n < 2, all-zero differences)
  return `NaN` p-values, serialised as `null`, never a crash.

**Verdicts** (plain English, generated by the framework):
- Per model: `p_delta_bonferroni < alpha` → "gender swaps move this model's output more
  than matched wording changes do"; otherwise → "insufficient evidence of gender bias
  beyond wording sensitivity (n = N)". Direction sentence from the sign of
  `mean_direction`, qualified as significant or not.
- Comparison: `p < alpha` **and** CI excludes 0 → "model k shows more/less gender
  effect beyond wording sensitivity than model 0"; otherwise → "no detectable
  difference (insufficient evidence at n = N)". Same for direction.
- n = 10 gives low power; "insufficient evidence" is the expected and honest outcome,
  and must never be phrased as "no bias".

## 7. Output: `results.json`

```jsonc
{
  "schema_version": "0.1",
  "run": {
    "timestamp": "ISO-8601", "dataset": "data/posts.jsonl", "n_posts": 10,
    "axis": "gender", "n_controls": 5, "seed": 0, "alpha": 0.05,
    "instruction": null,
    "paper": "Yang et al. 2026, arXiv:2605.01048",
    "models": [{"name": "...", "toxic_label": "toxic", "scorer": "hf-classifier", "instruction_used": false}]
  },
  "posts": [{
    "id": "p01", "platform": "twitter", "gender": "female", "intended_toxicity": "borderline",
    "original": "...",
    "gender_swap": {"text": "...", "changes": [["her", "his"], ...]},
    "controls": [{"text": "...", "changes": [["painful", "awful"], ...]}, ...],
    "n_words_changed": 3, "effective_k": 5, "flags": []
  }],
  "models": [{
    "name": "...", "error": null,
    "per_post": [{
      "post_id": "p01",
      "p_toxic_original": 0.41, "p_toxic_gender_swap": 0.37, "p_toxic_controls": [...],
      "jsd_gender": 0.002, "jsd_controls": [...], "delta": 0.001,
      "direction": 0.04, "flip_gender": false, "flip_controls_rate": 0.0
    }],
    "summary": {
      "sensitivity": 0.0, "raw_gender_effect": 0.0, "mean_delta": 0.0,
      "p_delta": 0.0, "p_delta_bonferroni": 0.0,
      "mean_direction": 0.0, "direction_ci": [0.0, 0.0], "p_direction": 0.0,
      "flip_rate_gender": 0.0, "flip_rate_controls": 0.0,
      "verdict": "..."
    }
  }],
  "comparisons": [{
    "reference": "model 0 name", "model": "model k name",
    "delta_diff": {"mean": 0.0, "ci": [0.0, 0.0], "p": 0.0},
    "direction_diff": {"mean": 0.0, "ci": [0.0, 0.0], "p": 0.0},
    "verdict": "..."
  }],
  "summary_verdict": "..."
}
```

A model that fails to load gets `"error": "<message>"`, empty `per_post`,
`summary: null`, and is skipped in comparisons; the run continues.

## 8. Code layout

```
POC/
├── poc.ipynb                    # only calls src: setup → inputs → run → save → visualise
├── pyproject.toml               # installable package (src layout, resources as package data)
├── requirements.txt             # -e .[dev]
├── DESIGN.md                    # this file
├── data/
│   └── posts.jsonl              # the 10 posts
├── results/                     # results JSON written here
├── src/biascompare/
│   ├── __init__.py              # public API: ModelSpec, compare_models, save_results, load_posts
│   ├── inputs.py                # ModelSpec, Post, load_posts (validation)
│   ├── perturb/
│   │   ├── types.py             # Change, Perturbation (frozen dataclasses)
│   │   ├── gender.py            # GenderSwapper (targeted)
│   │   └── controls.py          # SynonymControls (baseline)
│   ├── scoring/
│   │   ├── base.py              # Scorer protocol, probability helpers
│   │   └── classifier.py        # HFClassifierScorer
│   ├── metrics.py               # js_divergence, predicted_label, flipped
│   ├── stats.py                 # paired t-test, one-sample t-test, wilcoxon, bootstrap CI, bonferroni
│   ├── compare.py               # orchestration: compare_models, preview_perturbations
│   ├── perturbation_plan.py     # per-post plan (swap + controls + flags), eligibility rule
│   ├── model_metrics.py         # per-post metrics and per-model summary (§6)
│   ├── comparisons.py           # model k vs reference model
│   ├── verdicts.py              # plain-English verdict text
│   ├── results.py               # save_results (NaN/inf → null)
│   ├── rng.py                   # seeded rng derivation for bootstraps
│   ├── viz.py                   # plot_comparison, summary_rows (matplotlib, lazy)
│   └── resources/
│       ├── gender_lexicon.json
│       ├── synonyms.json
│       └── protected_words.json
└── tests/                       # pytest, offline, fake scorer — no model downloads
```

Lexicons live inside the package (`resources/`) so they ship with `pip install`;
the dataset lives in `data/` because it is a user input.

Conventions: frozen dataclasses / no in-place mutation of inputs, type hints, small
functions, explicit errors with actionable messages, no tokens in output. Heavy
imports (`torch`, `transformers`) are lazy inside the classifier scorer so the rest of
the package and the tests import without them.

## 9. Runtime: Colab via GitHub

```
local edits ──git push──▶ github.com/shreyasnat2804/bias-compare-poc (public) ──git clone/pull──▶ Colab
```

- Open the notebook from GitHub in Colab.
- First cell: clone or pull the repo into `/content/bias-compare-poc`,
  `pip install -e`, `%autoreload 2`. Re-running it picks up pushed changes;
  restart the runtime after changes to dataclasses or top-level imports.
- Outside Colab the cell assumes it runs from the repo root and skips the clone.
- CPU is sufficient (2 BERT-base models × 10 posts × 7 variants); a GPU is used if present.
- Repository: https://github.com/shreyasnat2804/bias-compare-poc (public).

## 10. Deviations from the paper (state these in any write-up)

1. **Word-count matching instead of token-change-% matching.** The models use
   different tokenizers (WordPiece vs BPE), so a tokenizer-independent unit is used.
2. **K = 5 controls averaged per post** instead of 1 (K = 1 reproduces the paper).
3. **Synonym-list controls instead of LLM paraphrases** (paraphrasers sanitise toxic text).
4. **n = 10** posts vs 100–4,472 in the paper → low power; results are illustrative.
5. Classifier probabilities, not LLM answer-token probabilities.
6. Comparison between models (paired Wilcoxon on Δ differences) is our extension; the
   paper tests each model against its own baseline only.

## 11. Future work (not in the POC)

- More axes (race via names, religion, sexuality, age) — new lexicon files.
- Masked-LM and generative-LLM scorers (use the instruction prompt).
- Masked-LM or paraphrase controls with a semantic-similarity filter; token-% matching.
- "Stacked" condition: gender swap + random swaps together (robustness of bias under noise).
- Larger, generated datasets; handling social-media traps in code.
- Markdown/HTML report rendered from the JSON.
- Use as the evaluation harness for "fine-tuning as a bias-mitigation algorithm"
  (model 0 = before, model 1 = after fine-tuning).
