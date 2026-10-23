# biascompare: framework plan

Status: Phase 1 implemented (see §11 for what is built and what is still to verify).
Items marked **planned** belong to later phases. The POC design record is
[`DESIGN.md`](../DESIGN.md); section references like "DESIGN §6" point there.

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

The framework keeps that contract and generalises it: a **task** defines an ordered
set of 2 to 4 toxicity levels, and every model, whatever its architecture, is wrapped in
an **adapter** whose only job is to turn a batch of texts into one distribution over
those levels per text.

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
                  one distribution over task levels per text
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
shared by all models. Decision: the framework is for **toxicity only, with 2 to 4
ordered levels**; no other label sets.

| Field | Meaning | Example | Status |
|-|-|-|-|
| `name` | Identifier | `toxicity-3` | built |
| `levels` | 2 to 4 levels, least to most toxic | `["non-toxic", "borderline", "toxic"]` | built |
| `toxic_from` | First level counted as toxic in the binary readout `p_toxic` | `toxic` | built |
| `instruction` | Natural-language question, used only by prompted adapters | `How toxic is the following post?` | built, unused until Phase 2 |
| `options` | Per level, the answer string an LLM is asked to produce | `{"non-toxic": "None", "borderline": "Some", "toxic": "Clear"}` | planned |
| `verbalizers` | Per level, words a masked LM may fill in | `{"non-toxic": ["fine"], "toxic": ["toxic"]}` | planned |
| `prompt_template` | How instruction, options and text are combined for LLMs | see §5 | planned |

Built-in tasks: `toxicity` (2 levels, the default), `toxicity-3` and `toxicity-4`
(DATA.md §3.1). The POC's run-level `instruction` argument now sets the task's
instruction.

### 3.2 Model adapters

One protocol, four implementations. "Uses prompt" means the adapter reads the task's
instruction and template; classifiers ignore them, exactly as in the POC.

| Adapter `type` | Models | How the distribution is obtained | Uses prompt |
|-|-|-|-|
| `hf-classifier` | BERT, RoBERTa, DeBERTa, DistilBERT with a classification head | Softmax (or per-head sigmoid) over the head's logits, mapped onto task labels with a `label_map` | No |
| `hf-masked-lm` | `bert-base-uncased`, `roberta-base`, ... with no task head | Logits at the `[MASK]` position, restricted to each label's verbalizer tokens, then normalised | Yes |
| `hf-causal-lm` | Llama, Mistral, Qwen, Gemma, ... run locally | Logits for each option's answer tokens after the rendered prompt, then normalised over the options | Yes |
| `api-llm` | Hosted LLMs | Token log-probabilities if the API returns them; otherwise the empirical distribution of N sampled answers | Yes |

The protocol as built in Phase 1 (`scoring/base.py`):

```python
class Scorer(Protocol):
    name: str
    scorer_type: str          # "hf-classifier", "hf-causal-lm", ...
    uses_instruction: bool
    # optional: metadata: dict, copied into run.models[] (revision, label mapping, ...)

    def score(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...  # one prob per level
    def close(self) -> None: ...

ScorerFactory = Callable[[ModelSpec, TaskSpec, str], Scorer]  # (spec, task, device)
```

Every returned distribution is checked (length equals the number of levels, finite,
non-negative, sums to 1); a model that fails the check is recorded as an error. Planned
for the LLM adapters: per-text diagnostics (option mass, invalid answer rate,
truncation) next to the probabilities.

Adapters are built by a registry keyed on `type` (`scoring/registry.py`), so the
orchestration code does not import any of them directly. `torch`, `transformers` and API
client libraries stay lazy imports inside the adapter that needs them; making each an
optional extra (`biascompare[hf]`, `biascompare[api]`) is planned with Phase 2.

### 3.3 Perturbations

The perturbation layer is already model-independent and stays that way.

- **Targeted swap**: a gender lexicon swap (DESIGN §5.1), looked up through an axis
  registry (`axes.py`): an axis is a lexicon file plus a swapper, so new axes (race via
  names, religion, age) are data, not code, where a word-level swap is valid. Only
  `gender` exists.
- **Controls**: two generators behind one interface (`perturb/factory.py`). `synonyms`
  is the POC's hand-written list (DESIGN §5.2), still the default so the POC reproduces
  exactly, but it only covers the sample posts. `masked-lm` (built in Phase 1,
  `perturb/mlm_controls.py`) works on arbitrary text: a helper masked LM proposes
  in-context replacements for content words, and a sentence-embedding similarity
  filter drops controls that drift from the original. Both helper models are separate
  from the models under test and are freed before any of those load. Known limit: a
  masked LM can propose antonyms ("great" for "awful"), which a similarity filter does
  not fully catch, so the threshold needs checking on real data (§11).
- **Only the text slot is perturbed.** For prompted models the instruction, options and
  system prompt are fixed text around the row's `text`; they are never swapped.

### 3.4 Metrics and statistics

The statistics in `stats.py`, `comparisons.py` and `verdicts.py` work on per-row
scalars (Δi, direction) and carried over unchanged. The per-row metrics were
generalised to K levels in Phase 1; for K = 2 every number is identical to the POC
(locked by a golden-file test, §10):

| Metric | POC (2 levels) | Framework (2 to 4 levels) |
|-|-|-|
| `jsd_*` | JSD over `[P(non-toxic), P(toxic)]` | JSD over the full distribution over levels |
| `flip_*` | `P(toxic) ≥ 0.5` differs | most probable level differs (ties go to the more toxic level) |
| `direction` | P(toxic) of the female-referencing version minus that of the male-referencing version | the same difference on the **severity score**, the expected level scaled to [0, 1], which equals P(toxic) for 2 levels |

Added in Phase 1:

- **Standardised effect size** `mean(Δ) / sd(Δ)` per model, scale-free (see §6).
- **Competence check**: if rows carry a gold `label`, each model's accuracy on the
  original texts is reported next to its bias numbers (macro-F1 is planned).
- **Run warnings**: excluded rows, unbalanced groups, low row count.

Planned: **adapter diagnostics** aggregated per model (mean option mass, invalid
answer rate, share of truncated rows).

## 4. Adapter details

### 4.1 `hf-classifier` (BERT family)

This is the POC's `HFClassifierScorer`, generalised in Phase 1.

- **Label mapping** (built). Model label names differ (`toxic`, `toxicity`,
  `LABEL_1`, `offensive`). A `label_map` lists, per level, the model labels whose
  softmax probabilities are summed into it; the lowest level may be left out and then
  receives every other label. Every model label must land in exactly one level and
  every level must get at least one, or the model fails with a message listing both
  sides. The POC's `toxic_label` stays as the 2-level shorthand. Rules in DATA.md §3.2.
- **Multi-label heads** (`problem_type == "multi_label_classification"`, e.g.
  toxic-bert's toxic / severe_toxic / insult / ... heads): one named head is read with
  a sigmoid, as in the POC, giving `[1 − p, p]`. Independent sigmoids do not form a
  distribution over 3 or 4 levels, so such models can only join 2-level runs.
- **Reproducibility** (built). Optional `revision` (a commit hash on the Hub); the
  resolved revision is written to the results, with the label mapping used.
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

- **Same levels, or no comparison.** All models in a run are scored over the same task
  levels; a scorer that returns a different number of probabilities is recorded as an
  error (built). This matters for the 3 and 4-level tasks: binary classifiers, and
  multi-label classifiers in particular, can only join 2-level runs. Planned for
  Phase 2: an LLM asked for 3 or 4 levels can have its answer distribution collapsed
  to 2 levels (summing the levels from `toxic_from` upwards) so it can be compared with
  binary classifiers in a 2-level run.
- **Δ is a within-model contrast, but JSD is scale-dependent.** Δ already subtracts
  each model's own wording sensitivity, which is what makes cross-model comparison
  meaningful at all. But JSD depends on how confident a model is: LLM option
  distributions are often very peaked (close to 0 or 1), so their JSDs are near 0 unless
  the answer flips, and near 1 when it does. A classifier with softer probabilities can
  show a larger mean Δ with the same underlying behaviour. The framework therefore
  reports, next to the raw Δ comparison:
  - the standardised effect size `mean(Δ) / sd(Δ)` per model (built),
  - flip-based metrics (gender flip rate vs. control flip rate), which do not depend on
    calibration (built).
- **Estimation noise.** Sampling-based distributions (§4.4) add Monte Carlo noise. It
  affects the targeted and control arms alike, so Δ stays roughly centred, but it
  inflates the variance of Δ and biases JSD upward. A model scored by sampling should
  not be compared with one scored from logits without saying so; the results will flag
  any comparison where the two models used different estimation modes (planned).
- **Competence.** A model that always answers "non-toxic" has Δ = 0 and would look
  unbiased. When gold labels are available, accuracy on the originals is reported next
  to the bias numbers (built) so a "no detectable bias" result from an incompetent
  model is visible as such.
- **The results say all of this.** A `comparability` block per comparison will list
  the two adapter types, estimation modes, and any of the warnings above, and the
  verdict text will mention it when the types differ (planned with Phase 2).

## 7. Configuration and entry points

A run is described by one JSON or YAML config file, so that it can be versioned and
rerun (built). The Python API keeps working for notebooks. A Phase 1 config:

```yaml
# configs/validation.yaml
dataset: ../data/validation/tweet_eval_offensive_gender.jsonl
task: toxicity            # built-in (toxicity, toxicity-3, toxicity-4) or a task block
n_controls: 5
seed: 0
controls:
  method: masked-lm
  min_similarity: 0.85

models:                   # the first model that scores successfully is the reference
  - name: unitary/toxic-bert            # multi-label: reads the "toxic" sigmoid head
    toxic_label: toxic
  - name: s-nlp/roberta_toxicity_classifier
    label_map: {toxic: [toxic], non-toxic: [neutral]}
```

LLM entries will sit in the same list once their adapters exist (planned):

```yaml
  - type: hf-causal-lm                  # Phase 2
    name: Qwen/Qwen2.5-7B-Instruct
    dtype: bfloat16
    use_chat_template: true
    scoring: first-token

  - type: api-llm                       # Phase 4
    provider: anthropic
    model: claude-opus-5-5
    estimation: sampling
    n_samples: 20
    api_key_env: ANTHROPIC_API_KEY
```

```python
from biascompare import load_config, run

results = run(load_config("configs/validation.yaml"))
```

On the command line, `biascompare run CONFIG` writes the results JSON (to `output`,
or `results/<config name>.json` next to the config) and prints the verdict and
warnings; `biascompare preview CONFIG` builds the perturbations only. The field
reference is in [`DATA.md` §3](./DATA.md#3-models-task-and-run-configuration).

## 8. Results schema

The v0.1 schema (DESIGN §7) is extended, never replaced: every v0.1 field keeps its
meaning.

**Version 0.2 (Phase 1, built)** adds `run.task`, `run.controls`, model `type`,
`revision` and label mapping in `run.models[]`, row `attributes`/`label`/`meta`,
control `similarity`, full per-level distributions per row, `effect_size`,
`n_eligible`, `accuracy`, `n_labeled`, and a top-level `warnings` list. The full list is
in DATA.md §5.

**Version 1.0 (planned, with the LLM adapters)** adds:

- `run.task`: answer options and the rendered template's hash.
- `run.models[]`: API model id and date, estimation mode.
- per-row `diagnostics` (option mass, invalid rate, truncated) and their per-model
  aggregates.
- `comparisons[]`: the `comparability` block from §6.

## 9. Package layout

Built in Phase 1 (new modules marked):

```
src/biascompare/
├── __init__.py            # compare_models, run, load_config, TaskSpec, ControlsSpec, ...
├── __main__.py, cli.py    # new: `biascompare run|preview CONFIG`
├── config.py              # new: RunConfig, JSON/YAML loading and validation, run()
├── tasks.py               # new: TaskSpec, built-in toxicity tasks (2 to 4 levels)
├── axes.py                # new: axis registry (lexicon per axis)
├── devices.py             # new: shared cuda/mps/cpu selection
├── inputs.py              # ModelSpec (type, label_map, revision, token_env, settings), Post (v1), loader
├── perturb/
│   ├── gender.py          # lexicon swapper
│   ├── controls.py        # synonym-list controls
│   ├── mlm_controls.py    # new: masked-LM controls and their HF helper models
│   └── factory.py         # new: ControlsSpec, build_control_generator
├── scoring/
│   ├── base.py            # Scorer protocol, level mapping helpers
│   ├── registry.py        # new: "type" -> adapter factory
│   └── classifier.py      # hf-classifier, generalised to task levels
├── perturbation_plan.py, model_metrics.py, compare.py
├── metrics.py, stats.py, comparisons.py, verdicts.py, rng.py, results.py, viz.py
└── resources/             # lexicon, synonyms, protected words, stopwords
configs/                   # new: example run configs
scripts/                   # new: build_validation_set.py
```

Planned: `scoring/hf_causal_lm.py`, `scoring/hf_masked_lm.py`, `scoring/api_llm.py`,
`scoring/prompts.py` (template rendering and validation), `scoring/cache.py` (on-disk
score cache), and CSV and Hugging Face dataset loaders.

## 10. Testing strategy

- **Offline by default** (built): fake scorers drive orchestration, metrics and
  statistics tests with no downloads.
- **Regression lock** (built): the POC's output on the sample data with the fake
  scorers is saved in `tests/golden/`. The current code must reproduce every value in
  it exactly, from both the v1 and the original v0.1 copy of the data.
- **Adapter integration tests** (built): tiny randomly initialised BERT and RoBERTa
  models are built from configs and saved locally, then loaded through
  `from_pretrained`, so the real `transformers` code paths (tokenisation, mask
  handling, pooling, label mapping, an end-to-end run with masked-LM controls) are
  tested with no network. Marked `integration`; deselect with
  `pytest -m "not integration"`.
- **API adapter tests** (planned) replay recorded responses from fixture files; no
  live calls in CI.
- **Property tests** for the prompt rules in §5 (planned): for every row, the rendered
  prompts of all arms differ only inside the text slot.

## 11. Roadmap

Each phase ends with the test suite green and a short update to this document.

| Phase | Scope | Done when |
|-|-|-|
| 0 | Fork, this plan, the data specification | Done |
| 1 | **Core generalisation.** 2 to 4-level distributions, `TaskSpec`, `label_map` for classifiers, v1 dataset schema with a v0.1 reader, config file loading and CLI, adapter registry. **Masked-LM controls** so real datasets get usable baselines | Built. Golden-file regression passes. Still to verify on Colab: the 200-row TweetEval set (`scripts/build_validation_set.py`, `configs/validation.yaml`) runs with under 5% of rows flagged `no_controls`, and the `min_similarity` default holds up |
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

Decided:

- **Task scope:** toxicity only, with 2 to 4 ordered levels (§3.1).
- **Config format:** both JSON and YAML are accepted (`pyyaml` is a dependency).

Still open:

1. **Which LLMs come first?** Local open-weight models (Phase 2) or hosted APIs
   (Phase 4)? The order above assumes local first because logits make the cleanest
   comparison and need no API budget.
2. **Cross-type comparisons.** Should a comparison between a classifier and an LLM be a
   first-class verdict, or always reported with the comparability warning attached?
3. **Data source.** Will real social media data be used? That decides how much of the
   privacy and licensing guidance in DATA.md §2.6 becomes tooling.
4. **Repository visibility.** The POC is public so Colab can clone it without a token.
   If this repo stays private, the example notebook needs a token-based clone step.
