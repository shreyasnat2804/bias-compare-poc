# biascompare

A comparative, counterfactual bias testing framework for text models: fine-tuned
**BERT-style classifiers** and **LLMs**, side by side. Given a set of models, a dataset
and a task definition, it measures how much each model's output moves when a protected
attribute in the input is swapped (he ↔ she), **beyond** how much it moves for a matched,
irrelevant wording change, and then compares every model against a reference model.

The method follows Yang, Levy, Goldberg & Wallace (2026), *"Compared to What? Baselines
and Metrics for Counterfactual Prompting"* (arXiv:2605.01048): a demographic swap that
changes a model's output is only evidence of bias if it changes the output more than a
baseline perturbation of the same size does.

## Status: Phase 1 of the plan

This repository is the official framework. It was forked from
[`bias-compare-poc`](https://github.com/shreyasnat2804/bias-compare-poc) at commit
`7d4020d`, which showed that two Hugging Face toxicity classifiers can be compared this
way. Phase 1 has generalised the POC's core; LLM support starts in Phase 2.

What works now:

- **Toxicity tasks with 2 to 4 ordered levels** (`toxicity`, `toxicity-3`,
  `toxicity-4`, or your own level names).
- **Hugging Face classifiers** (BERT, RoBERTa, ...) with a `label_map` from the
  model's own labels to the task's levels, and pinned revisions.
- **The v1 dataset format** (`attributes`, optional gold `label`, `meta`), with the
  POC's format still accepted.
- **Masked-LM controls** that work on any text, alongside the POC's synonym list.
- **Run config files** (JSON or YAML) and a `biascompare` command.
- Per-model effect size, accuracy against gold labels, and data-quality warnings.

With the POC's settings, the framework reproduces the POC's numbers exactly (locked by
a golden-file test).

| Document | What it covers |
|-|-|
| [`docs/PLAN.md`](./docs/PLAN.md) | The plan: how one framework takes in both BERT-style models and LLMs, the components, the comparability rules, and the phased roadmap with what is built |
| [`docs/DATA.md`](./docs/DATA.md) | What data the framework expects: the dataset format and content rules, task levels, label mapping, run config fields, resource files, and the output |
| [`DESIGN.md`](./DESIGN.md) | The POC design record: method, statistics, and the v0.1 results schema |

## Quick start

```bash
pip install -e .
biascompare preview configs/poc.yaml   # perturbations only, no models loaded
biascompare run configs/poc.yaml       # scores both classifiers, writes results/poc.json
```

Or from Python:

```python
from biascompare import ModelSpec, compare_models, save_results

results = compare_models(
    [
        ModelSpec("unitary/toxic-bert", toxic_label="toxic"),
        ModelSpec("s-nlp/roberta_toxicity_classifier", label_map={"toxic": ["toxic"], "non-toxic": ["neutral"]}),
    ],
    "data/posts.jsonl",
    task="toxicity",
    controls="masked-lm",   # use this for any dataset other than the sample posts
)
save_results(results, "results/results.json")
```

Before using your own dataset, read [`docs/DATA.md`](./docs/DATA.md). In particular,
use `controls: masked-lm`: the default synonym list only knows the words in the sample
posts.

## Validating Phase 1 on real data (Colab)

The masked-LM controls need two small helper models from the Hugging Face Hub, so this
check runs on Colab (or any machine with Hub access). From a clone of this repository:

```bash
pip install -e .
python scripts/build_validation_set.py      # 200 gender-balanced TweetEval tweets
biascompare preview configs/validation.yaml # want fewer than 5% of rows flagged no_controls
biascompare run configs/validation.yaml
```

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
src/biascompare/     the package (tasks, perturbations, scoring, statistics, config, CLI)
data/posts.jsonl     10 hand-written sample posts (v1 format)
configs/             example run configs
scripts/             build_validation_set.py
docs/                the framework plan and the data specification
DESIGN.md            the POC design record
tests/               offline pytest suite, golden POC results, tiny-model integration tests
```
