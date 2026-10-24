# biascompare

A comparative, counterfactual bias testing framework for text models: fine-tuned
**BERT-style classifiers** and **LLMs** from Hugging Face, side by side. Given a set of
models and a dataset of posts, it measures how much each model's output moves when a
protected attribute in a post is swapped (she ↔ he, "deaf cousin" ↔ "cousin"),
**beyond** how much it moves for a matched, irrelevant wording change, and then
compares the models with each other.

The method follows Yang, Levy, Goldberg & Wallace (2026), *"Compared to What? Baselines
and Metrics for Counterfactual Prompting"* (arXiv:2605.01048): a demographic swap that
changes a model's output is only evidence of bias if it changes the output more than a
baseline perturbation of the same size does.

## Status

This is the testing framework for the KQV study (fine-tuning and prompt engineering,
alone and combined, on BERT-family content moderation classifiers). It was forked from
[`bias-compare-poc`](https://github.com/shreyasnat2804/bias-compare-poc) at commit
`7d4020d`. With the POC's settings it reproduces the POC's numbers exactly (locked by a
golden-file test).

**Input**: Hugging Face models (Hub ids or fine-tuned checkpoints in Drive), a dataset
of posts tagged with swaps and control synonyms, and, from the data side, a prompts
file whose few-shot examples carry their bias signatures.
**Output**: per bias axis (gender, disability), each model's bias beyond wording
sensitivity, a ranking, and Holm-corrected pairwise comparisons, as tables, plots and
one JSON file.

What it supports:

- **The M0 to M3 design**: the same model as several variants via `id`, each mapped to
  a prompt from the data side's prompts file (instruction, or instruction plus
  few-shot examples). The framework checks that no few-shot example is an evaluated
  post.
- **Hugging Face classifiers** (BERT, RoBERTa, ModernBERT, ...) with label mapping,
  and **Hugging Face LLMs** scored on their answer-option probabilities.
- **Gender and disability axes** in one run (custom axes too), with **tagged
  multi-word swaps** and **tagged control synonyms** per post.
- **2 to 4 toxicity levels**, accuracy against gold labels, and data-quality warnings.

| Document | What it covers |
|-|-|
| [`notebooks/bias_study.ipynb`](./notebooks/bias_study.ipynb) | The Colab notebook for the study |
| [`docs/DATA.md`](./docs/DATA.md) | What data and settings the framework expects (the tag format is §2.3), and what it outputs |
| [`docs/PLAN.md`](./docs/PLAN.md) | How the framework maps onto the proposal, how classifiers and LLMs share one test, and status |
| [`DESIGN.md`](./DESIGN.md) | The POC design record: method and statistics |

## Running the study on Colab

Open `notebooks/bias_study.ipynb` in Colab (File, Open notebook, GitHub tab, this
repository; pick the branch that holds the framework). Run the cells top to bottom:
the first one clones the repository and installs it, the model cell defines M0 to M3,
the prompts file and the dataset are inputs you point it at. It runs as is on the
placeholders `data/sample_prompts.yaml` and `data/sample_study.jsonl` (20 hand-written
tagged posts), with two public classifiers standing in for your models. Set a Colab secret `HF_TOKEN` only for gated or private
models.

The same run from a terminal:

```bash
pip install -e .
biascompare preview configs/study.yaml   # swaps and controls only, no models loaded
biascompare run configs/study.yaml       # writes results/study.json, prints each axis's ranking
```

Or from Python:

```python
from biascompare import ModelSpec, compare_axes, load_prompts

prompts = load_prompts("data/sample_prompts.yaml")  # the data side's prompts file
base, tuned = "s-nlp/roberta_toxicity_classifier", "/content/drive/MyDrive/kqv/m2"
models = [
    ModelSpec(base, id="M0", prompt=prompts["instruction"]),
    ModelSpec(base, id="M1", prompt=prompts["few_shot"]),
    ModelSpec(tuned, id="M2", prompt=prompts["instruction"]),
    ModelSpec(tuned, id="M3", prompt=prompts["few_shot"]),
]
report = compare_axes(models, "data/sample_study.jsonl", axes=["gender", "disability"], controls="tagged")
```

## Comparing two LLMs (Qwen3-8B vs Llama-3.1-8B)

`notebooks/llm_comparison.ipynb` runs the same test on two 8B Hugging Face LLMs, each
scored on its probability for the answer options of the data side's `llm` prompt. It
needs a GPU runtime (on a T4 the models load in 4-bit, on an L4 or A100 in 16-bit) and,
for Llama, an accepted licence on the Hub plus a Colab secret `HF_TOKEN`. It previews
each model's exact prompt from the tokenizer alone before downloading any weights. From a
terminal on a GPU machine: `biascompare run configs/llm_8b.yaml`.

## Local development

```bash
pip install -e .[dev]
pytest -q
```

Tests are fully offline: fake scorers, plus integration tests on tiny randomly
initialised models built locally, so no model is downloaded. Skip the integration tests
with `pytest -m "not integration"`. Viz and integration tests are skipped automatically
if `matplotlib` or `torch` is not installed.

## Layout

```
src/biascompare/          the package
notebooks/bias_study.ipynb  the Colab notebook for the study (M0 to M3)
notebooks/llm_comparison.ipynb  Qwen3-8B vs Llama-3.1-8B on Colab
data/sample_study.jsonl   20 hand-written tagged posts (gender and disability)
data/sample_prompts.yaml  placeholder prompts file (+ sample_few_shot.jsonl examples)
data/posts.jsonl          the POC's 10 sample posts
configs/                  study.yaml, llm_8b.yaml, poc.yaml, validation.yaml
scripts/                  build_validation_set.py (TweetEval gender set)
docs/                     the data specification and the plan
DESIGN.md                 the POC design record
tests/                    offline pytest suite, golden POC results, tiny-model integration tests
```
