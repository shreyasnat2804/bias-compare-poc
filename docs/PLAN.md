# biascompare: framework plan and status

Status: the testing framework for the KQV study is built (§11). Items marked
**planned** are not. The POC design record is [`DESIGN.md`](../DESIGN.md); "DESIGN §6"
points there. The data and configuration reference is [`DATA.md`](./DATA.md).

## 1. What the framework is for

The proposal (Natarajan, Ethiraj, Venkatesan, "Auditing Social Bias in Language Models:
Metrics & a Category-Aware Framework") asks how fine-tuning and prompt engineering,
alone and combined, change bias in a BERT-family content moderation classifier,
measured as counterfactual sensitivity beyond paraphrase-induced variation. Its
contribution (3) is a reusable auditing framework that accepts both encoder
classifiers and LLMs. This repository is that framework.

**Input**: Hugging Face models, run on Google Colab: the variants M0 to M3, and
optionally LLMs; a dataset of posts tagged with swaps and control synonyms; and from the
data side, the prompts and few-shot examples (the examples carry their bias
signatures). **Output**: the models' relative bias on each axis.

Division of work: the data side delivers the tagged posts and the prompts file with its
few-shot examples; the testing framework (this repository) loads and checks them,
applies each variant's prompt, runs the counterfactual test and compares the models.

| Proposal | Framework |
|-|-|
| Prompts and labelled few-shot examples | A prompts file from the data side (DATA.md §3.4); examples are data rows with attributes and tagged swaps, which templates can show (for example each example next to its counterfactual) |
| M0: pretrained BERT-family classifier, instruction prompt | `ModelSpec(base, id="M0", prompt=prompts["instruction"])` |
| M1: M0 plus prompt engineering (labelled few-shot examples) | `ModelSpec(base, id="M1", prompt=prompts["few_shot"])`, same model, different prompt |
| M2: M0 fully fine-tuned | `ModelSpec(<fine-tuned checkpoint>, id="M2", prompt=prompts["instruction"])`, a Hub id or a Drive folder |
| M3: M2 plus prompt engineering | `ModelSpec(<fine-tuned checkpoint>, id="M3", prompt=prompts["few_shot"])` |
| LLMs in the same audit | `ModelSpec(..., type="hf-causal-lm")` |
| Gender and disability axes | Built-in axes `gender` and `disability`, run together with `compare_axes` |
| Multi-word swaps of all co-referring words and modifiers | Per-row tagged `swaps`, phrase pairs, counted in word edits (DATA.md §2.3) |
| Controls changing the same number of words with neutral synonyms | Per-row tagged `synonyms`, `controls: tagged` |
| 8B LLM tagging, 300 verified posts | The tagging step writes the row format in DATA.md §2.3; the framework validates and previews it |
| Δ = JSD(swap) minus mean JSD(controls); one-sided paired t-test | Unchanged from the POC (DESIGN §6), Bonferroni across models |
| Signed direction statistic | Severity of the target group's version (women, disabled people) minus the other's |
| Pairwise model comparisons | All pairs (6 for M0 to M3), paired Wilcoxon on per-post differences, bootstrap CIs, Holm-corrected; plus a ranking by mean Δ |
| Runs on Colab | `notebooks/bias_study.ipynb` |

What carries over unchanged from the POC: the method and statistics (DESIGN §1, §6),
perturbations built once per post and shared by every model, a failing model recorded
as an error without stopping the run, "insufficient evidence" never phrased as "no
bias", and offline tests. A golden-file test locks the POC's numbers.

## 2. The one idea that lets classifiers and LLMs share a test

Every statistic is computed from one thing: for each input text, a **probability
distribution over a fixed set of toxicity levels**. A **task** fixes 2 to 4 ordered
levels, and every model, whatever its architecture, is wrapped in an **adapter** whose
only job is to turn texts into one distribution over those levels per text.

```
                     ┌───────────────┐
 tagged posts ──────▶│ perturbation  │──▶ original, targeted swap, K controls
                     │ plan per axis │    (identical for every model)
                     └───────┬───────┘
                             │ texts (each model wraps them in its own prompt)
                 ┌───────────┴────────────┐
                 ▼                        ▼
           hf-classifier            hf-causal-lm
        (BERT + head, M0..M3)     (local LLM, answer-option
                                      probabilities)
                 └───────────┬────────────┘
                             ▼
             one distribution over task levels per text
                             │
                             ▼
       Δ, direction, per-model tests, pairwise comparisons, ranking
```

Adding a kind of model means writing one adapter; the statistics do not change.

## 3. Components

### 3.1 Task (built)

Toxicity only, with 2 to 4 ordered levels (built-in `toxicity`, `toxicity-3`,
`toxicity-4`, or custom names); DATA.md §3.1. JSD uses the full distribution, flips the
most probable level, direction the severity score (expected level in [0, 1], equal to
P(toxic) for 2 levels).

### 3.2 Model adapters

| `type` | Models | How the distribution is obtained | Status |
|-|-|-|-|
| `hf-classifier` | BERT, RoBERTa, DeBERTa, ModernBERT ... with a classification head (M0 to M3) | Softmax (or one sigmoid head) over the head's logits, mapped onto levels with `label_map` | built |
| `hf-causal-lm` | Llama, Qwen, Mistral, Gemma ... run on Colab | Next-token probabilities of each answer option after the rendered prompt | built |
| `hf-masked-lm` | Plain pretrained BERT with a cloze prompt | Logits at the mask, restricted to each level's verbalizer words | planned |
| hosted-API LLMs | Models reached through an API | Log-probabilities, or sampled answers | not needed: every model in the study is a Hugging Face model |

Adapters are chosen by `type` through a registry (`scoring/registry.py`). Every
returned distribution is checked (one probability per level, sums to 1); a model that
fails is recorded as an error. Adapters may report diagnostics (truncated inputs, LLM
option mass), which the results aggregate.

### 3.3 Axes and perturbations (built)

- **Axes** (`axes.py`): two groups, a target group for direction, display labels for
  verdicts, an optional lexicon. Built in: `gender` (lexicon or tags) and `disability`
  (tags). Custom axes in code or config, no code change (DATA.md §3.3).
- **Targeted swap**: from the row's tags when it has them for the axis (multi-word
  phrase pairs, every occurrence, longest first, case kept), else the axis lexicon.
- **Controls**: `tagged` (row synonyms only, the study's setting), `synonyms` (shared
  list plus row synonyms, the POC default), `masked-lm` (helper masked LM with a
  similarity filter, for untagged data). Controls change as many words as the swap
  (word edit distance) and never touch swap words, lexicon words or protected toxic
  words.
- **Only the post is perturbed.** Prompts are fixed text around it (§5).

### 3.4 Metrics, tests and comparisons (built)

Per post: `jsd_swap`, `jsd_controls`, `delta`, `direction`, flips. Per model:
sensitivity, raw swap effect, mean Δ with its one-sided t-test and Bonferroni across
models, mean direction with bootstrap CI and t-test, flip rates, standardised effect
size, prediction shares (the label-free competence check, with a warning when a model
gives every post the same answer), accuracy if gold labels are given, truncation count,
LLM option mass, verdict. Between
models: every pair (or every model against the first), Wilcoxon on the per-post
differences in Δ and in direction, bootstrap CIs, **Holm-adjusted** p-values used in
the verdicts, and a ranking by mean Δ. Data-quality warnings per axis.

## 4. Adapter details

### 4.1 `hf-classifier` (built)

- `toxic_label` or `label_map` maps the model's labels onto the levels; multi-label
  (sigmoid) models can only join 2-level runs (DATA.md §3.2).
- With a prompt, the classifier scores the rendered prompt (instruction, few-shot
  examples, post). Inputs longer than `max_length` are cut **from the left**, so the
  post is kept; the number of truncated rows is reported and warned about. This is how
  M1 and M3 differ from M0 and M2. Note for the write-up: a classifier fine-tuned on
  raw posts has not seen instructions or examples in its input, so prompt engineering
  on an encoder classifier changes its input distribution as well as the information
  it gets; that is part of what the M1 and M3 comparisons measure.
- `revision` pins a Hub commit; the resolved revision and the label mapping are saved.

### 4.2 `hf-masked-lm` (planned)

A plain pretrained BERT scored with a cloze template (`{text} This post is [MASK].`)
and verbalizer words per level. Useful if M0 is meant to be a pretrained encoder
without a classification head; not needed when M0 is a toxicity classifier.

### 4.3 `hf-causal-lm` (built)

- The prompt (instruction, examples, post) is followed by "Answer with exactly one of:
  ...", and wrapped in the tokenizer's chat template (with an optional system message)
  when it has one.
- `first-token` scoring: softmax over each option's first answer token, renormalised
  over the options; the options must start with different tokens. `sequence` scoring:
  the probability of each full option string, for options that share a first token.
- The probability mass put on the options before renormalising is the per-text
  `option_mass`; a low mean means the model is not answering in the requested format,
  and the results warn.
- Batches are left-padded with explicit position ids. Without them, a short text's
  scores changed with the batch it was in, enough to flip its answer in testing.
- `dtype` (`float16` on a T4) and `quantization` (`4bit`, `8bit` via bitsandbytes)
  for larger models; one model is loaded at a time and freed before the next.

## 5. Prompt rules that keep the test valid (built)

1. The instruction, examples, options, template and system message are identical for
   the original, the swap and every control of a post; only the `{text}` slot changes.
   `PromptSpec.render` is a pure function of the post, so this holds by construction.
2. Few-shot examples come from the data side and are fixed for the whole run, not
   chosen per post. (If examples are ever selected per post, they must be selected
   from the original and reused for its swap and controls.) No example may be an
   evaluated post or its swap: the run checks this and refuses to start otherwise.
3. Option order is fixed for the whole run, so any position bias affects every arm
   equally and cancels in Δ.
4. The instruction must not mention the protected attribute ("Is this post sexist?"
   would change what is measured).
5. The rendered prompt for the first post is saved per model (`prompt_example`).

## 6. Comparing a classifier with an LLM

- **Same levels.** All models in a run are scored over the same task levels; binary
  classifiers need a 2-level task. Planned: collapse an LLM's 3 or 4-level answer to 2
  levels so it can join a binary run.
- **Scale.** Δ subtracts each model's own wording sensitivity, but JSD depends on
  calibration: LLM option distributions are often peaked, so their JSDs are near 0
  unless the answer flips. Compare classifiers with LLMs on the effect size and the
  flip rates, not only on raw Δ.
- **Competence.** A model that always answers "non-toxic" has Δ = 0 because it ignores
  the post. Each model's `prediction_shares` show this without any labels, and the run
  warns about it. The evaluated posts need no toxicity labels (DATA.md §2.2), so no
  labeller's bias enters the measurement; optional human labels add accuracy.
- **Format.** An LLM with low option mass is not following the answer format; its
  distribution over the options is less meaningful.
- Planned: a `comparability` block per comparison stating the adapter types and these
  caveats.

## 7. Configuration and entry points (built)

- **Colab notebook** `notebooks/bias_study.ipynb`: setup (clone, install, `HF_TOKEN`
  from Colab secrets, optional Drive), models and prompts, data preview, run, ranking
  and pairwise tables, plots, results saved (to Drive too, if mounted).
- **Python**: `compare_axes(models, dataset, axes=[...])` for several axes in one run
  (each model loaded once), `compare_models(..., axis=...)` for one.
- **Config files and CLI**: `biascompare preview|run CONFIG` with JSON or YAML;
  `configs/study.yaml` is the study's setup. Field reference: DATA.md §3.5.

## 8. Results schema

Schema `0.3` (DATA.md §5): per axis, posts, per-post metrics, per-model summaries,
pairwise comparisons, ranking, verdict and warnings. It renamed the POC's
gender-specific fields to axis-neutral ones; every value is unchanged (locked by the
golden test). Planned: the `comparability` block (§6).

## 9. Package layout

```
src/biascompare/
├── __init__.py            # public API
├── compare.py             # compare_models, compare_axes, preview_perturbations, axis_results
├── config.py, cli.py      # run configs, `biascompare run|preview`
├── tasks.py               # TaskSpec and built-in toxicity tasks (2 to 4 levels)
├── axes.py                # AxisSpec, built-in gender and disability axes
├── prompts.py             # PromptSpec, Example: rendering prompts and few-shot examples
├── prompt_files.py        # loading the data side's prompts file and example rows
├── inputs.py              # ModelSpec, Post (tags), dataset loading and validation
├── perturb/               # lexicon swaps, tagged swaps, synonym / tagged / masked-LM controls
├── perturbation_plan.py   # per-axis plan, flags, eligibility
├── scoring/               # Scorer protocol, label mapping, registry, hf-classifier, hf-causal-lm
├── model_metrics.py, comparisons.py, stats.py, verdicts.py, metrics.py, rng.py
├── results.py, viz.py     # JSON output, plots and tables
└── resources/             # lexicon, synonyms, protected words, stopwords
notebooks/bias_study.ipynb # the Colab notebook for the study
configs/                   # study, POC and validation configs
data/                      # posts.jsonl (POC), sample_study.jsonl, sample_prompts.yaml, sample_few_shot.jsonl
scripts/                   # build_validation_set.py
```

## 10. Testing

- **Offline by default**: fake scorers drive orchestration, metrics, statistics,
  prompts, tagged swaps, axes, comparisons and configs.
- **Regression lock**: the POC's results on its sample data are stored in
  `tests/golden/` and must be reproduced exactly, from both data formats.
- **Integration** (marker `integration`): tiny randomly initialised BERT, RoBERTa,
  GPT-2 and Llama models are built and saved locally, then loaded through
  `from_pretrained`, so the real `transformers` code paths run with no download:
  tokenisation, prompts and left truncation, masked-LM controls, chat templates,
  answer tokenisation, batch-independent LLM scoring, and end-to-end runs.
- The notebook's code was executed end to end against such tiny models (5 models, 2
  axes, 10 pairwise comparisons per axis).
- **Not yet done anywhere**: a run with the real Hugging Face models. This repository's
  build environment cannot reach the Hub, so the first real run is on Colab.

## 11. Roadmap and status

| Step | Scope | Status |
|-|-|-|
| 0 | Fork, plan, data specification | done |
| 1 | Task levels, label mapping, v1 rows, configs and CLI, adapter registry, masked-LM controls | done |
| Study | Model ids and per-model prompts, disability axis, tagged swaps and synonyms, all-pairs Holm-corrected comparisons and ranking, multi-axis runs, Colab notebook | done |
| 2 | `hf-causal-lm` for Hugging Face LLMs | done |
| Next | First real run on Colab with the sample data, then with the tagged ETHOS / ToxicCommons posts; check prompted-classifier truncation and LLM option mass | to do |
| 3 | `hf-masked-lm` (cloze-prompted BERT) | planned, if needed |
| Later | 2-level collapse for LLMs, comparability block, Markdown/HTML report from the JSON, more axes | planned |
| Dropped | Hosted-API LLMs | not needed (all models are Hugging Face models) |

## 12. Open questions

1. **Prompts file and example rows.** The data side's prompts and few-shot examples
   should follow DATA.md §3.4. Which bias-signature fields should the examples carry
   (attributes and swaps are supported; anything else fits in `meta`), and should the
   examples be shown with their counterfactual (`few_shot_pairs` in the sample)?
2. **Tag format.** The tagging step should emit DATA.md §2.3 as is. Is the
   per-occurrence behaviour (a phrase replaces every occurrence) right for your posts,
   or do some need occurrence-level tags?
3. **Prompting the classifiers.** The framework prepends the instruction and examples
   to the classifier's input (§4.1). If M0 is instead a pretrained encoder without a
   classification head, it needs the planned cloze adapter (§4.2).
4. **Repository.** The framework still lives on a branch of `bias-compare-poc`; the
   notebook clones that branch. A separate `bias-compare` repository needs to be
   created on GitHub (the integration here cannot create repositories).
5. **Data source and licences** for the ETHOS and ToxicCommons subsets (DATA.md §2.7).
