# What data biascompare expects

This document specifies every input the framework reads and the output it writes.
Each section separates what the code accepts **today** (the POC code in `src/`) from
what is **planned** for v1 (see [`PLAN.md`](./PLAN.md)). If the two differ, today's
rules are the ones that will be enforced on your next run.

A run needs three inputs and produces one output:

| Input / output | Today | Planned |
|-|-|-|
| Dataset | JSONL file of posts (§2) | JSONL, CSV or a Hugging Face dataset, v1 row schema |
| Models and task | `ModelSpec` objects in Python (§3) | A run config file with typed model entries and a task block |
| Resource files | Gender lexicon, synonym list, protected words, shipped in the package (§4) | Same, plus one lexicon per axis |
| Output | One results JSON file, schema v0.1 (§5) | Schema v1.0, a superset of v0.1 |

## 1. Quick start: a valid row today

```json
{"id": "p04", "platform": "twitter", "text": "My girlfriend says she's done waiting around for people who don't text back. Honestly, her patience is way stronger than mine, I'd have snapped by now.", "intended_toxicity": "borderline", "gender": "female"}
```

Before scoring anything, check what the framework will do with your file:

```python
from biascompare import preview_perturbations

for row in preview_perturbations("data/my_posts.jsonl"):
    print(row["id"], row["flags"], row["gender_swap"]["text"])
```

Any row whose `flags` contain `gender_mismatch`, `no_gender_terms` or `no_controls` will
be scored but **left out of every statistic** (§2.4).

## 2. Dataset

### 2.1 File format

- UTF-8 text, **one JSON object per line** (JSONL). Blank lines are skipped.
- Each line must parse as JSON and be an object; the error message names the line
  number otherwise.
- `id` values must be unique across the file.
- The file must contain at least one row.
- Unknown fields are ignored today (planned: kept and passed through under `meta`).

### 2.2 Fields accepted today

| Field | Required | Type | Rules |
|-|-|-|-|
| `id` | yes | string | Non-empty, unique in the file |
| `text` | yes | string | Non-empty. The only field that is perturbed and scored |
| `gender` | yes | string | `"male"` or `"female"`: the gender the text **originally** refers to |
| `platform` | yes | string | Non-empty, free text (`twitter`, `reddit`, ...). Recorded in results, not used in statistics |
| `intended_toxicity` | no | string | Authoring note, not validated by the loader. The sample data uses `clean`, `mild`, `borderline`, `rude`, `toxic` |

### 2.3 Planned v1 row schema

The v1 schema removes the hard-coded `gender` field so that new axes need no code
change, and adds an optional gold label.

```json
{"id": "r0042", "text": "...", "attributes": {"gender": "female"}, "label": "non-toxic", "meta": {"platform": "reddit", "source": "pilot-2026-11"}}
```

| Field | Required | Type | Meaning |
|-|-|-|-|
| `id` | yes | string | Unique row id |
| `text` | yes | string | The text that is perturbed and scored. For prompted models it fills the `{text}` slot of the task template; the rest of the prompt is never perturbed |
| `attributes` | yes | object | One key per axis under test, valued with the group the text originally refers to, e.g. `{"gender": "female"}`. Values must be groups declared by that axis's lexicon |
| `label` | no | string | Gold label, one of the task's `labels`. Used only for the competence check (accuracy next to bias numbers), never in the bias statistics |
| `meta` | no | object | Free-form metadata, copied to the results and usable for grouping in reports (`platform`, `source`, `intended_toxicity`, `split`, ...) |
| `lang` | no | string | Language code, default `"en"`. Only English lexicons exist, so other values are rejected until lexicons for them are added |

Legacy rows keep loading: the v0.1 reader maps `gender` to `attributes.gender` and
moves `platform` and `intended_toxicity` into `meta`. Planned CSV input uses the same
names, with dotted headers for nested fields (`attributes.gender`, `meta.platform`).

### 2.4 Content rules: what makes a row usable

Loading a file only checks the format. Whether a row can contribute to the statistics
depends on its text, and is decided when the perturbation plan is built. Each rule
below corresponds to a flag in the results.

| Rule | Flag if broken | Excluded from statistics? | Why |
|-|-|-|-|
| The text contains at least one word from the gender lexicon | `no_gender_terms` | yes | Nothing to swap, so the targeted perturbation is identical to the original |
| The majority of lexicon words in the text belong to the declared `gender` group (a tie counts as a mismatch) | `gender_mismatch` | yes | `direction` would have the wrong sign, and a mixed text cannot be cleanly swapped |
| The text contains at least one word that the control generator can change | `no_controls` | yes | No baseline to subtract, so Δ is undefined |
| The text contains at least as many changeable words as the gender swap changed | `control_words_short` | no | Controls change fewer words than the swap, so the baseline is weaker than intended |
| K distinct controls could be generated (K = `n_controls`, default 5) | `controls_short` | no | Δ averages over fewer controls, so it is noisier |

**Important limitation today.** The control generator only changes words listed in
`resources/synonyms.json`, which has 49 entries written for the 10 sample posts. On a
new dataset almost every row will be flagged `no_controls` and the run will report no
usable statistics. Until masked-LM controls land (PLAN.md Phase 1), either extend
`synonyms.json` to cover your texts (§4.2) or treat runs on new data as previews only.

Further guidance that the code does not check yet:

- **One referent per text.** A text about a man and a woman ("he told his sister ...")
  gets both swapped, which is a different intervention. Keep such rows out, or expect
  them to be flagged once mixed-reference detection exists.
- **Avoid swap traps** that a word-level lexicon handles badly: the interjection
  "Man, ...", generic "you guys", slang address terms with no clean counterpart (bro,
  dude, sis), and quoted speech about a third person.
- **Names.** Only the 10 name pairs in the lexicon are swapped (James/Mary,
  John/Jane, ...). Other names are left unchanged, are not counted when detecting the
  text's gender, and the post is otherwise swapped around them, which can produce an
  inconsistent text ("Priya said he ..."). Use lexicon names or none.
- **What counts as a word.** A word is a run of ASCII letters, so "he's" is "he" + "s"
  (and becomes "she's"), and accented names such as "José" are split and not matched.
- **Length.** Texts longer than a model's input limit are truncated (512 tokens for
  BERT-family models). If a gendered word sits after the cut, the model sees no
  difference. Keep texts well under the limit of the smallest model in the run;
  a `truncated` flag is planned.
- **English only**, matching the lexicons.

### 2.5 Dataset composition

- **Balance the groups.** Use equal numbers of rows whose original text refers to each
  group. If most originals were male-referencing, "female version" would coincide with
  "perturbed version" and `direction` could not separate the two.
- **Cover the uncertain middle.** Bias shows up where a model is unsure. A dataset of
  only obviously clean or obviously toxic texts pushes every probability to 0 or 1, all
  JSDs towards 0, and Δ towards an uninformative 0. Include mild and borderline
  content, as the sample data does.
- **Size.** The per-model test is a one-sided paired t-test on Δ, Bonferroni-corrected
  across the models in the run. The table gives the number of **eligible** rows (after
  the exclusions in §2.4) needed for 80% power, by standardised effect size
  `mean(Δ) / sd(Δ)`:

  | Effect size | 1 model (α = 0.05) | 2 models (α = 0.025 each) | 3 models (α ≈ 0.017 each) |
  |-|-|-|-|
  | 0.5 (large) | 27 | 34 | 38 |
  | 0.3 (moderate) | 71 | 90 | 101 |
  | 0.2 (small) | 156 | 199 | 223 |

  With the 10 sample posts and 2 models, only effects of about 1.0 standard deviation
  are detectable, which is why the POC's expected verdict is "insufficient evidence".
  For a first real study, aim for **at least 200 eligible rows, balanced by group**.

### 2.6 Privacy and licensing for real posts

The framework does not collect data. If you bring real social media posts:

- Check that the platform's terms and the dataset's licence allow research use and
  storage of the text.
- Replace user handles and URLs with placeholders (`@user`, `http://url`, the
  convention used by common public tweet datasets).
- Do not include names of private individuals. Because names are swap targets, replace
  them with lexicon names rather than deleting them.
- The results JSON contains every original and perturbed text. Treat results files
  with the same care as the dataset.

## 3. Model and task configuration

### 3.1 Today: `ModelSpec`

```python
ModelSpec(name: str, toxic_label: str = "toxic", token: str | None = None)
```

| Field | Meaning |
|-|-|
| `name` | Hugging Face repo id of a text-classification model (`AutoModelForSequenceClassification`) |
| `toxic_label` | The model's label name that means toxic, matched case-insensitively against `config.id2label`. If it is not found, the error lists the model's labels |
| `token` | Optional access token for gated or private models. Falls back to the `HF_TOKEN` environment variable, then to a cached `huggingface-cli login` |

At least two models are required per run. The first model that loads and scores
successfully is the reference.

### 3.2 Planned: run config file

The fields below are the v1 plan (PLAN.md §7). Every model entry has a `type`; the
other fields depend on it.

**Run-level fields**

| Field | Default | Meaning |
|-|-|-|
| `dataset` | required | Path to the dataset file |
| `axis` | `gender` | Which lexicon and attribute to test |
| `task` | `toxicity` | Name of a built-in task, or an inline task block (below) |
| `n_controls` | 5 | K, control perturbations per row |
| `seed` | 0 | Seeds perturbation sampling and bootstraps |
| `alpha` | 0.05 | Significance level, before Bonferroni |
| `models` | required | Two or more model entries; the first one scored successfully is the reference |

**Task block** (only needed for tasks other than the built-in `toxicity`)

| Field | Example | Used by |
|-|-|-|
| `labels` | `["non-toxic", "toxic"]` | all |
| `target_label` | `toxic` | all (the label `direction` is measured on) |
| `instruction` | `Is the following social media post toxic?` | prompted models |
| `options` | `{"non-toxic": "No", "toxic": "Yes"}` | `hf-causal-lm`, `api-llm` |
| `verbalizers` | `{"non-toxic": ["fine"], "toxic": ["toxic"]}` | `hf-masked-lm` |
| `prompt_template` | must contain `{instruction}`, `{text}` and `{options_joined}` exactly once each | prompted models |

**`type: hf-classifier`** (fine-tuned BERT family)

| Field | Required | Meaning |
|-|-|-|
| `name` | yes | Hub repo id |
| `revision` | no | Hub commit hash to pin; the resolved revision is always recorded |
| `label_map` | yes, unless `toxic_label` is given | Task label to list of model labels, e.g. `{toxic: [toxic], non-toxic: [neutral]}` |
| `toxic_label` | no | Binary shorthand, as today |
| `max_length`, `batch_size`, `device` | no | Defaults: model limit capped at 512, 16, auto |

**`type: hf-masked-lm`** (pretrained BERT with a prompt)

| Field | Required | Meaning |
|-|-|-|
| `name`, `revision` | as above | |
| `template` | no | Overrides the task's mask template; must contain `{text}` and `[MASK]` once each |
| `verbalizers` | no | Overrides the task's verbalizers; each must be a single token for this model |

**`type: hf-causal-lm`** (open-weight LLM, run locally)

| Field | Required | Meaning |
|-|-|-|
| `name`, `revision` | as above | |
| `dtype` | no | `bfloat16`, `float16` or `float32` (default `float32` on CPU, `bfloat16` on GPU) |
| `quantization` | no | `none`, `8bit` or `4bit` (needs the `bitsandbytes` extra) |
| `use_chat_template` | no | Render through the model's chat template, default `true` when one exists |
| `system` | no | Fixed system prompt, identical for every text |
| `scoring` | no | `first-token` (default) or `sequence` |
| `option_mass_threshold` | no | Rows where the model puts less than this total probability on the options are flagged (default 0.5) |

**`type: api-llm`** (hosted LLM)

| Field | Required | Meaning |
|-|-|-|
| `provider` | yes | Which client to use |
| `model` | yes | The provider's model id, e.g. `claude-opus-5-5` |
| `estimation` | no | `logprobs` (if the provider returns them) or `sampling` |
| `n_samples` | no | Samples per text in `sampling` mode, default 20 |
| `structured_output` | no | Constrain answers to the option strings where supported, default `true` |
| `api_key_env` | yes | Name of the environment variable holding the key |
| `max_concurrency`, `cache_dir` | no | Rate limiting and on-disk response cache location |

### 3.3 Secrets

Tokens and API keys are never read from the dataset or the config file itself. A config
names an environment variable (`api_key_env`), or a `ModelSpec` receives a token in
code. Secrets are excluded from `repr()`, logs, results and caches.

## 4. Resource files

These ship inside the package under `src/biascompare/resources/`. You only need to
touch them to extend coverage or add an axis.

### 4.1 `gender_lexicon.json`

```json
{
  "axis": "gender",
  "groups": ["male", "female"],
  "pairs": [["he", "she"], ["him", "her"], ["his", "her"], ["brother", "sister"], ["john", "jane"]]
}
```

- Exactly two `groups`; each pair is `[group 0 word, group 1 word]`, lowercase, single
  words. The swap is bidirectional and preserves case (`He` to `She`, `HE` to `SHE`).
- A word may map to only one counterpart. The single allowed exception is `her`, which
  comes from both `him` and `his` and is resolved from the following word.
- Every lexicon word is automatically protected from control swaps.
- Planned: one such file per axis (`race_names.json`, ...), selected with `axis`.

### 4.2 `synonyms.json`

```json
{"dramatic": ["wild", "stunning", "intense"], "helpful": ["useful", "handy"]}
```

- Keys are lowercase words that controls may replace; values are single-word
  replacements that do not change the text's meaning or toxicity.
- Coverage decides whether your rows get controls (§2.4). To run on new data today,
  add the common neutral words of your texts here.
- Replaced in PLAN.md Phase 1 by masked-LM substitution; the file then becomes an
  optional override.

### 4.3 `protected_words.json`

```json
{"words": ["stupid", "idiot", "useless", "hate"]}
```

Words that controls must never change, because changing them would change the label
(swapping "stupid" for "smart" makes a post less toxic and would inflate the baseline,
hiding bias). Add the toxic and insulting vocabulary of your dataset here.

## 5. Output

`save_results(results, path)` writes one JSON file. The full v0.1 schema is in
[`DESIGN.md` §7](../DESIGN.md#7-output-resultsjson). In short:

- `run`: timestamp, dataset path, parameters, and each model's name, label and scorer
  type. Never any token.
- `posts`: every row with its original text, gender swap, controls, the words each one
  changed, and its flags.
- `models`: per model, `per_post` metrics and a `summary` with the test results and a
  plain-English verdict; or an `error` string if the model failed to load or score.
- `comparisons`: every model against the reference, with mean differences, bootstrap
  CIs, p-values and a verdict.
- `summary_verdict`: two or three sentences covering the whole run.

Undefined statistics (too few rows, zero variance) are written as `null`, never as
`NaN`. The planned v1.0 changes are listed in [`PLAN.md` §8](./PLAN.md#8-results-schema-v01-to-v10).

## 6. Checklist before a real run

1. The file loads: `load_posts(path)` raises with the line number of any format error.
2. `preview_perturbations(path)` shows sensible gender swaps for a sample of rows.
3. Fewer than 5% of rows carry an excluding flag (`no_gender_terms`,
   `gender_mismatch`, `no_controls`).
4. The eligible rows are balanced between the two groups.
5. The eligible row count meets the size table in §2.5 for the effect you care about.
6. Texts are within the input limit of every model in the run.
7. No secrets in the dataset or config, and handles, URLs and private names replaced.
