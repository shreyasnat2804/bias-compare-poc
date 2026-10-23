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

## Status: planning

This repository is the official framework. It was forked from
[`bias-compare-poc`](https://github.com/shreyasnat2804/bias-compare-poc) at commit
`7d4020d`, which showed that two Hugging Face toxicity classifiers can be compared this
way. The code under `src/` is the POC code, carried over unchanged as the starting
point. It currently supports Hugging Face text classifiers and the gender axis only.

| Document | What it covers |
|-|-|
| [`docs/PLAN.md`](./docs/PLAN.md) | The plan: how one framework takes in both BERT-style models and LLMs, the components, the comparability rules, and the phased roadmap |
| [`docs/DATA.md`](./docs/DATA.md) | What data the framework expects: the dataset format and content rules, model and task configuration, resource files, and the output |
| [`DESIGN.md`](./DESIGN.md) | The POC design record: method, statistics, and the v0.1 results schema. Still accurate for the code as it is today |

## What works today

```python
from biascompare import ModelSpec, compare_models, save_results

results = compare_models(
    [
        ModelSpec("unitary/toxic-bert", toxic_label="toxic"),
        ModelSpec("s-nlp/roberta_toxicity_classifier", toxic_label="toxic"),
    ],
    "data/posts.jsonl",
    n_controls=5,
    seed=0,
)
save_results(results, "results/results.json")
```

`preview_perturbations("data/posts.jsonl")` shows the gender swaps and controls that a
run would score, without loading any model. Before using your own dataset, read
[`docs/DATA.md`](./docs/DATA.md): the current control generator only knows the words in
the sample posts, so most new posts will be flagged `no_controls` until that is replaced
(Phase 1 of the plan).

## Local development

```bash
pip install -e .[dev]
pytest -q
```

Tests are fully offline (fake scorers, no model downloads or network access). Viz tests
are skipped automatically if `matplotlib` is not installed.

## Layout

```
src/biascompare/     the package (perturbations, scoring, statistics, results)
data/posts.jsonl     10 hand-written sample posts (the POC dataset)
docs/                the framework plan and the data specification
DESIGN.md            the POC design record
tests/               offline pytest suite
```
