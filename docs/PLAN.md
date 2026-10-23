# biascompare: framework plan

Status: draft for review. Nothing in this document is implemented yet unless it says
"today". The POC design record is [`DESIGN.md`](../DESIGN.md); section references like
"DESIGN §6" point there.

## 1. Goal

Turn the POC into a framework that can compare the bias of **any mix of text models**
on the same counterfactual test:

- fine-tuned BERT-family classifiers (`unitary/toxic-bert`, RoBERTa, DeBERTa, ...),
- BERT-family masked language models used with a prompt,
- open-weight LLMs run locally through `transformers`,
- hosted LLMs reached through an API.

The POC already proved the method on two classifiers. What it lacks is a way to plug in
models that are not classifiers, a dataset format that is not tied to one axis, a
control generator that works on arbitrary text, and a way to scale past 10 posts.

What carries over unchanged:

- The method (DESIGN §1): targeted swap vs. matched-magnitude controls, Δi, the
  one-sided paired t-test with Bonferroni across models, and model k vs. model 0
  comparisons with bootstrap CIs and Wilcoxon p-values (DESIGN §6).
- Perturbations are generated once per row and shared by every model (DESIGN §5).
- A model that fails to load or score is recorded as an error and the run continues
  (DESIGN §7).
- Verdict wording: a non-significant result is "insufficient evidence", never "no bias".
- Conventions: frozen dataclasses, lazy heavy imports, tokens never written anywhere,
  offline tests.

## 2. The one idea that lets BERT and LLMs share a test

Every statistic in the POC is computed from one thing: for each input text, a
**probability distribution over a fixed set of outcomes**. The POC hard-codes that set
as `[P(non-toxic), P(toxic)]`.

The framework keeps that contract and generalises it: a **task** defines the label set,
and every model, whatever its architecture, is wrapped in an **adapter** whose only job
is to turn a batch of texts into one distribution over the task's labels per text.

```
                     ┌───────────────┐
 dataset rows ──────▶│ perturbation  │──▶ original, targeted swap, K controls
                     │ plan (shared) │    (identical for every model)
                     └───────┬───────┘
                             │ texts
          ┌──────────────────┼──────────────────┬──────────────────┐
          ▼                  ▼                  ▼                  ▼
   hf-classifier       hf-masked-lm       hf-causal-lm         api-llm
   (BERT + head)      (BERT + prompt)    (local LLM logits)  (hosted LLM)
          │                  │                  │                  │
          └──────────────────┴────────┬─────────┴──────────────────┘
                                      ▼
                  one distribution over task labels per text
                                      │
                                      ▼
                 metrics, statistics, comparisons, verdicts
                   (model-type agnostic, as in the POC)
```

Everything downstream of the adapters stays model-type agnostic. Adding a new kind of
model means writing one adapter, not touching the statistics.

## 3. Components

### 3.1 Task

A task says what question every model is answering. It is defined once per run and
shared by all models.

| Field | Meaning | Toxicity example |
|-|-|-|
| `name` | Identifier | `toxicity` |
| `labels` | Ordered outcome set, K ≥ 2 | `["non-toxic", "toxic"]` |
| `target_label` | The label whose probability `direction` is measured on | `toxic` |
| `instruction` | Natural-language question, used only by prompted adapters | `Is the following social media post toxic?` |
| `options` | Per label, the answer string an LLM is asked to produce | `{"non-toxic": "No", "toxic": "Yes"}` |
| `verbalizers` | Per label, words a masked LM may fill in | `{"non-toxic": ["fine", "harmless"], "toxic": ["toxic", "offensive"]}` |
| `prompt_template` | How instruction, options and text are combined for LLMs | see §5 |

A built-in `toxicity` task ships with the package so the common case needs no task
config. The POC's run-level `instruction` argument moves into the task.

### 3.2 Model adapters

One protocol, four implementations. "Uses prompt" means the adapter reads the task's
instruction and template; classifiers ignore them, exactly as in the POC.

| Adapter `type` | Models | How the distribution is obtained | Uses prompt |
|-|-|-|-|
| `hf-classifier` | BERT, RoBERTa, DeBERTa, DistilBERT with a classification head | Softmax (or per-head sigmoid) over the head's logits, mapped onto task labels with a `label_map` | No |
| `hf-masked-lm` | `bert-base-uncased`, `roberta-base`, ... with no task head | Logits at the `[MASK]` position, restricted to each label's verbalizer tokens, then normalised | Yes |
| `hf-causal-lm` | Llama, Mistral, Qwen, Gemma, ... run locally | Logits for each option's answer tokens after the rendered prompt, then normalised over the options | Yes |
| `api-llm` | Hosted LLMs | Token log-probabilities if the API returns them; otherwise the empirical distribution of N sampled answers | Yes |

Protocol sketch (generalises `scoring/base.py` today):

```python
class Scorer(Protocol):
    name: str
    scorer_type: str          # "hf-classifier", "hf-causal-lm", ...
    uses_instruction: bool

    def score(self, texts: Sequence[str]) -> list[ScoredText]: ...
    def close(self) -> None: ...

@dataclass(frozen=True)
class ScoredText:
    probs: tuple[float, ...]          # one per task label, sums to 1
    diagnostics: Mapping[str, float]  # e.g. option_mass, invalid_rate, truncated
```

Adapters are built by a registry keyed on `type`, so the orchestration code does not
import any of them directly. `torch`, `transformers` and API client libraries stay lazy
imports inside the adapter that needs them, and each is an optional extra
(`pip install biascompare[hf]`, `biascompare[api]`).

### 3.3 Perturbations

The perturbation layer is already model-independent and stays that way.

- **Targeted swap**: today a gender lexicon swap (DESIGN §5.1). Generalised to an axis
  registry: an axis is a lexicon file plus a swapper, so new axes (race via names,
  religion, age) are data, not code, where a word-level swap is valid.
- **Controls**: today a hand-written synonym list (DESIGN §5.2) that only covers the
  sample posts. This is the main blocker for real datasets (see DATA.md §2.4) and is
  replaced in Phase 1 by masked-LM substitution with a semantic-similarity filter,
  using small helper models that are separate from the models under test.
- **Only the text slot is perturbed.** For prompted models the instruction, options and
  system prompt are fixed text around the row's `text`; they are never swapped.

### 3.4 Metrics and statistics

The statistics in `stats.py`, `comparisons.py` and `verdicts.py` already work on
per-row scalars (Δi, direction) and carry over as they are. The per-row metrics need
small, mechanical generalisations:

| Metric | Today (K = 2 only) | Framework (any K) |
|-|-|-|
| `jsd_*` | JSD over `[P(non-toxic), P(toxic)]` | JSD over the full K-label distribution (`js_divergence` already handles any K) |
| `flip_*` | `P(toxic) ≥ 0.5` differs | argmax label differs |
| `direction` | P(toxic) of the female-referencing version minus P(toxic) of the male-referencing version | same, on `task.target_label`, with the axis declaring which group is "first" |

Added in the framework:

- **Standardised effect size** `mean(Δ) / sd(Δ)` per model, scale-free (see §6).
- **Competence check**: if rows carry an optional gold `label`, each model's accuracy
  and macro-F1 on the original texts are reported next to its bias numbers.
- **Adapter diagnostics** aggregated per model: mean option mass, invalid answer rate,
  share of truncated rows.

## 4. Adapter details

### 4.1 `hf-classifier` (BERT family)

This is the POC's `HFClassifierScorer`, generalised.

- **Label mapping.** Model label names differ (`toxic`, `toxicity`, `LABEL_1`,
  `offensive`). The config maps each task label to one or more model labels:
  `label_map: {"toxic": ["toxic"], "non-toxic": ["neutral"]}`. For a softmax head,
  probabilities of model labels mapped to the same task label are summed; unmapped
  labels are an error unless `unmapped: ignore` is set, in which case the rest is
  renormalised. The POC's `toxic_label` stays as a shorthand for the binary case.
- **Multi-label heads** (`problem_type == "multi_label_classification"`, e.g.
  toxic-bert's toxic / severe_toxic / insult / ... heads): the task's target label is
  read from one named head with a sigmoid, as in the POC, giving `[1 − p, p]`. Other
  heads are ignored and the choice is recorded.
- **Reproducibility.** Optional `revision` (a commit hash on the Hub). The resolved
  revision is always written to the results.
- **Truncation.** Inputs longer than the model's max length are truncated (512 tokens
  for BERT). If a swapped word falls after the cut, the model sees identical inputs for
  the original and the swap. Planned: detect this per row and flag it `truncated`.

### 4.2 `hf-masked-lm` (prompted BERT)

Lets a plain pretrained BERT be tested without a fine-tuned head, and is the cheapest
"prompted" model to compare against an LLM.

- Template with one mask slot:
  `"{text} Overall, this post is [MASK]."` (the adapter substitutes the tokenizer's own
  mask token).
- Score of a label = mean logit (or log-sum-exp) over its verbalizer tokens at the mask
  position; softmax across labels.
- Verbalizers must each be a single token in the model's vocabulary; this is validated
  at load time with a message listing the offending words.

### 4.3 `hf-causal-lm` (open-weight LLMs, local)

This is the setting the paper itself used: the model's probabilities over the answer
options.

- **Prompt rendering.** The task's template is filled with the row's text and rendered
  through the model's chat template when it has one (`use_chat_template: true`), with
  an optional fixed `system` prompt. The rendered prompt ends where the answer starts.
- **Option scoring.**
  - `first-token` (default): softmax over the logits of each option's first token at
    the answer position. Requires options whose first tokens differ after tokenization
    ("Yes" / "No", "A" / "B"); validated at load time.
  - `sequence`: sum of token log-probabilities of each full option continuation, then
    normalised across options. Slower (one forward pass per option) but works for
    multi-token options.
- **Option mass.** Before renormalising, the total probability the model put on the
  option tokens is recorded as `option_mass`. A low value means the model was not
  following the answer format (for example starting with "Sure,"), and the
  distribution over options is then less meaningful. Rows below a threshold are
  flagged, and the per-model mean is reported.
- **Batching.** Causal LMs need left padding with an attention mask so the answer
  position lines up across a batch; this is the easiest thing to get wrong and gets
  its own test.
- **Memory.** `dtype: bfloat16 | float16 | float32` and optional 4-bit / 8-bit loading
  (`bitsandbytes`, an optional extra). A 7B to 8B model in bf16 needs about 16 GB of GPU
  memory. As in the POC, one model is loaded at a time and freed before the next.
- **Determinism.** Logit scoring involves no sampling, so a run is reproducible up to
  GPU numerics.

### 4.4 `api-llm` (hosted LLMs)

Hosted APIs differ in what they expose, so the adapter supports two estimation modes
and always records which one produced the numbers.

- **`logprobs`**: when the API returns token log-probabilities for the first answer
  token, use them exactly as `first-token` scoring in §4.3. Many APIs return only the
  top-k alternatives; an option missing from the top k gets a floor probability and the
  row's `option_mass` reflects what was seen.
- **`sampling`**: when no log-probabilities are available, call the model N times per
  text (default N = 20), parse each answer into an option, and use the empirical
  distribution with add-one smoothing so no probability is exactly 0. Unparseable
  answers are counted as `invalid_rate`, not silently dropped. Where the API supports
  structured output (an enum of the option strings), it is used to cut invalid answers
  to near zero. At the time of writing, the Claude Messages API returns no token
  log-probabilities, and current Claude models only sample at their default settings
  (API version `2026-10-21` rejects `temperature`, `top_p` and `top_k` outright), so
  Claude models run in `sampling` mode with no temperature control.
- **Cost and limits.** Sampling multiplies calls: rows × (2 + K controls) × N. A
  `dry_run` prints the call count and an estimated cost before anything is sent.
  Requests are rate limited, retried with backoff, and every response is written to an
  on-disk cache keyed by (provider, model, prompt hash, text, sample index) so a rerun
  or a crash costs nothing extra.
- **Keys.** The config names an environment variable (`api_key_env: ANTHROPIC_API_KEY`);
  the key itself never appears in config, results, logs or the cache.
- **Model drift.** Hosted models can change under the same name. The results record
  the model id string and the run date; comparisons across runs on different dates
  should say so.

## 5. Prompt rules that keep the test valid

The counterfactual design only works if the swap and the controls are the **only**
differences between the texts a model sees. For prompted adapters that means:

1. The instruction, options, system prompt and template are identical for the original,
   the targeted swap and every control of a row. Only the `{text}` slot changes.
2. Option order is fixed for the whole run. Any position bias then hits every arm
   equally and cancels in Δ. (Counterbalanced option orders are a later extension.)
3. The instruction must not mention the protected attribute (no "Is this post sexist?"
   when testing gender bias, since that changes what is being measured).
4. The text is inserted verbatim between clear delimiters, and the template is checked
   to contain exactly one `{text}` slot.
5. The rendered template's hash is written to the results so two runs can be checked
   for prompt equality.

Default toxicity template:

```
{instruction}

Post:
"""
{text}
"""

Answer with exactly one word: {options_joined}.
Answer:
```

## 6. Comparing a BERT classifier with an LLM

Mixing model types in one comparison is the point of the framework, but some numbers
travel across types better than others.

- **Same task, or no comparison.** All models in a run must answer over the same label
  set. The run refuses to compare models whose adapters were given different labels.
- **Δ is a within-model contrast, but JSD is scale-dependent.** Δ already subtracts
  each model's own wording sensitivity, which is what makes cross-model comparison
  meaningful at all. But JSD depends on how confident a model is: LLM option
  distributions are often very peaked (close to 0 or 1), so their JSDs are near 0 unless
  the answer flips, and near 1 when it does. A classifier with softer probabilities can
  show a larger mean Δ with the same underlying behaviour. The framework therefore
  reports, next to the raw Δ comparison:
  - the standardised effect size `mean(Δ) / sd(Δ)` per model,
  - flip-based metrics (gender flip rate vs. control flip rate), which do not depend on
    calibration.
- **Estimation noise.** Sampling-based distributions (§4.4) add Monte Carlo noise. It
  affects the targeted and control arms alike, so Δ stays roughly centred, but it
  inflates the variance of Δ and biases JSD upward. A model scored by sampling should
  not be compared with one scored from logits without saying so; the results flag
  any comparison where the two models used different estimation modes.
- **Competence.** A model that always answers "non-toxic" has Δ = 0 and would look
  unbiased. When gold labels are available, accuracy on the originals is reported next
  to the bias numbers so a "no detectable bias" result from an incompetent model is
  visible as such.
- **The results say all of this.** A `comparability` block per comparison lists the
  two adapter types, estimation modes, and any of the warnings above, and the verdict
  text mentions it when the types differ.

## 7. Configuration and entry points

A run is described by one config file, so that it can be versioned and rerun. The
Python API keeps working for notebooks.

```yaml
# runs/toxicity_gender.yaml
dataset: data/posts.jsonl
axis: gender
task: toxicity            # built-in; or an inline task block
n_controls: 5
seed: 0
alpha: 0.05

models:                   # the first model that scores successfully is the reference
  - type: hf-classifier
    name: unitary/toxic-bert
    label_map: {toxic: [toxic]}          # multi-label head: read the "toxic" head

  - type: hf-classifier
    name: s-nlp/roberta_toxicity_classifier
    label_map: {toxic: [toxic], non-toxic: [neutral]}

  - type: hf-causal-lm
    name: Qwen/Qwen2.5-7B-Instruct
    dtype: bfloat16
    use_chat_template: true
    scoring: first-token

  - type: api-llm
    provider: anthropic
    model: claude-opus-5-5
    estimation: sampling
    n_samples: 20
    api_key_env: ANTHROPIC_API_KEY
```

```python
from biascompare import load_config, run

results = run(load_config("runs/toxicity_gender.yaml"))
```

A thin command-line entry point (`biascompare run runs/toxicity_gender.yaml`) wraps the
same call and writes the results JSON next to the config. The full field reference for
every model type is in [`DATA.md` §3](./DATA.md#3-model-and-task-configuration).

## 8. Results schema: v0.1 to v1.0

The v0.1 schema (DESIGN §7) is extended, not replaced. Main changes:

- `run.task`: labels, target label, instruction, template hash.
- `run.models[]`: `type`, resolved `revision` or API model id, estimation mode, the
  adapter config with secrets removed.
- `per_post[]` (renamed `per_row[]`): full distributions (`p_original`,
  `p_targeted`, `p_controls`) keyed by label, with `p_target_*` kept for convenience,
  plus a `diagnostics` object (option mass, invalid rate, truncated).
- `summary`: adds `effect_size`, `accuracy` (when gold labels exist), aggregated
  diagnostics.
- `comparisons[]`: adds the `comparability` block from §6.
- `schema_version` becomes `"1.0"`; a reader for v0.1 files is kept so old POC results
  still load into the viz helpers.

## 9. Proposed package layout

```
src/biascompare/
├── __init__.py            # run, load_config, compare_models (kept), save_results, ...
├── config.py              # RunConfig, TaskSpec, per-type model specs; YAML/JSON loading
├── data/
│   ├── schema.py          # Row dataclass, validation, v0.1 legacy mapping
│   └── loaders.py         # JSONL today; CSV and Hugging Face datasets later
├── tasks/
│   └── toxicity.py        # the built-in toxicity task
├── perturb/               # as today, plus an axis registry and masked-LM controls
├── scoring/
│   ├── base.py            # Scorer protocol, ScoredText, label mapping helpers
│   ├── registry.py        # "type" -> adapter factory
│   ├── hf_classifier.py   # today's classifier.py, generalised
│   ├── hf_masked_lm.py
│   ├── hf_causal_lm.py
│   ├── api_llm.py
│   ├── prompts.py         # template rendering and validation
│   └── cache.py           # on-disk score cache
├── metrics.py, stats.py, model_metrics.py, comparisons.py, verdicts.py
├── results.py             # save/load, v0.1 reader
├── viz.py
└── cli.py
```

## 10. Testing strategy

- **Offline by default**, as today: fake scorers drive orchestration, metrics and
  statistics tests with no downloads.
- **Regression lock.** Before any refactor, the current POC output on
  `data/posts.jsonl` with a fake scorer is saved as a golden file. Phase 1 must
  reproduce it exactly, so generalising to K labels cannot silently change the numbers.
- **Adapter integration tests** against tiny random-weight Hub models
  (`hf-internal-testing/*`), behind a `pytest -m integration` marker so the default run
  stays offline.
- **API adapter tests** replay recorded responses from fixture files; no live calls in
  CI.
- **Property tests** for the prompt rules in §5: for every row, the rendered prompts of
  all arms differ only inside the text slot.

## 11. Roadmap

Each phase ends with the test suite green and a short update to this document.

| Phase | Scope | Done when |
|-|-|-|
| 0 | Fork, this plan, the data specification | Reviewed and agreed (this commit) |
| 1 | **Core generalisation.** K-label distributions, `TaskSpec`, `label_map` for classifiers, v1 dataset schema with a v0.1 reader, config file loading, adapter registry. **Masked-LM controls** so real datasets get usable baselines | Golden-file regression passes; a 200-row public dataset runs with under 5% of rows flagged `no_controls` |
| 2 | **`hf-causal-lm`.** Prompt rendering and validation, first-token and sequence scoring, option mass, left-padded batching, dtype and quantisation options | A BERT classifier and a 7B instruct model compared in one run (on Colab via 4-bit loading) |
| 3 | **`hf-masked-lm`.** Templates and verbalizers | Prompted BERT runs in the same comparison |
| 4 | **`api-llm`.** Logprobs and sampling modes, structured output, cache, rate limits, dry run with cost estimate | A hosted model joins the comparison; a rerun hits the cache for every call |
| 5 | **Scale and axes.** CSV and Hugging Face dataset loaders, a dataset validation report, a power-analysis helper, a second axis | Data validation report is the first cell of the example notebook |
| 6 | **Reporting.** Markdown/HTML report rendered from the results JSON | A report is produced from any v1.0 results file |

Later: counterbalanced option orders, multiple prompt templates per model (prompt
sensitivity as its own axis), the "stacked" condition (swap plus random wording
changes), and using the framework as the evaluation harness for fine-tuning as a
bias-mitigation method (model 0 = before, model 1 = after).

## 12. Open questions

These need a decision before or during Phase 1.

1. **Which LLMs come first?** Local open-weight models (Phase 2) or hosted APIs
   (Phase 4)? The order above assumes local first because logits make the cleanest
   comparison and need no API budget.
2. **Task scope.** Toxicity only for v1, or general text classification (sentiment,
   hate speech, hiring screens) from the start? The design supports both; the choice
   decides how much effort goes into the built-in task library.
3. **Config format.** YAML (friendlier, one extra dependency) or JSON (no dependency)?
4. **Cross-type comparisons.** Should a comparison between a classifier and an LLM be a
   first-class verdict, or always reported with the comparability warning attached?
5. **Data source.** Will real social media data be used? That decides how much of the
   privacy and licensing guidance in DATA.md §2.6 becomes tooling.
6. **Repository visibility.** The POC is public so Colab can clone it without a token.
   If this repo stays private, the example notebook needs a token-based clone step.
