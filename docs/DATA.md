# What data biascompare expects

This document specifies every input the framework reads and the output it writes, as
of Phase 1 (see [`PLAN.md`](./PLAN.md) §11). Items marked **planned** belong to later
phases and are not accepted yet.

| Input / output | Format |
|-|-|
| Dataset | JSONL file, one row per line, in the v1 row format (§2.2) or the POC's v0.1 format (§2.3) |
| Run config | JSON or YAML file (§3.3), or the same arguments passed to `compare_models` in Python |
| Resource files | Gender lexicon, synonym list, protected words and stopwords, shipped in the package (§4) |
| Output | One results JSON file, schema version `0.2` (§5) |

## 1. Quick start

A valid row:

```json
{"id": "p04", "text": "My girlfriend says she's done waiting around for people who don't text back.", "attributes": {"gender": "female"}, "meta": {"platform": "twitter"}}
```

Check what the framework will do with a dataset before scoring anything:

```bash
biascompare preview configs/poc.yaml
```

or in Python:

```python
from biascompare import preview_perturbations

for row in preview_perturbations("data/my_posts.jsonl", controls="masked-lm"):
    print(row["id"], row["flags"], row["gender_swap"]["text"])
```

Any row whose `flags` contain `gender_mismatch`, `no_gender_terms` or `no_controls` is
scored but **left out of every statistic** (§2.4).

## 2. Dataset

### 2.1 File format

- UTF-8 text, **one JSON object per line** (JSONL). Blank lines are skipped.
- Each line must parse as JSON and be an object; the error message names the line
  number otherwise.
- The format is decided per line: a row with an `attributes` field is read as v1,
  any other row as v0.1. A file may mix both.
- `id` values must be unique across the file, and the file must contain at least one
  row.

### 2.2 v1 rows

```json
{"id": "r0042", "text": "...", "attributes": {"gender": "female"}, "label": "non-toxic", "meta": {"platform": "reddit", "source": "pilot-2026-11"}}
```

| Field | Required | Type | Meaning |
|-|-|-|-|
| `id` | yes | string | Unique row id |
| `text` | yes | string | The text that is perturbed and scored. For prompted models (planned) it fills the `{text}` slot of the prompt; the rest of the prompt is never perturbed |
| `attributes` | yes | object | One key per axis, valued with the group the text **originally** refers to, e.g. `{"gender": "female"}`. The run's axis must be present, and its value must be one of that axis's groups (`male` or `female` for `gender`) |
| `label` | no | string | Gold toxicity level, one of the run's task levels (§3.1), matched case-insensitively. Used only for the competence check (each model's accuracy on the original texts), never in the bias statistics |
| `meta` | no | object | Free-form metadata, copied to the results (`platform`, `source`, `split`, ...) |
| `lang` | no | string | Language code, default `"en"`. Only English lexicons exist, so any other value is rejected |

Any other top-level field is an error (it is usually a typo); put extra information
under `meta`.

### 2.3 v0.1 rows (the POC format, still accepted)

```json
{"id": "p04", "platform": "twitter", "text": "...", "intended_toxicity": "borderline", "gender": "female"}
```

| Field | Required | Becomes |
|-|-|-|
| `id`, `text` | yes | `id`, `text` |
| `gender` | yes, `"male"` or `"female"` | `attributes.gender` |
| `platform` | yes, non-empty | `meta.platform` |
| `intended_toxicity` | no | `meta.intended_toxicity` (an authoring note, not a gold label) |
| any other field | no | copied into `meta` |

### 2.4 Content rules: what makes a row usable

Loading a file only checks the format. Whether a row can contribute to the statistics
depends on its text, and is decided when the perturbation plan is built. Each rule
below corresponds to a flag in the results.

| Rule | Flag if broken | Excluded from statistics? | Why |
|-|-|-|-|
| The text contains at least one word from the gender lexicon | `no_gender_terms` | yes | Nothing to swap, so the targeted perturbation is identical to the original |
| The majority of lexicon words in the text belong to the declared group (a tie counts as a mismatch) | `gender_mismatch` | yes | `direction` would have the wrong sign, and a mixed text cannot be cleanly swapped |
| At least one word can be changed by the control generator | `no_controls` | yes | No baseline to subtract, so Δ is undefined |
| At least as many words can be changed as the gender swap changed | `control_words_short` | no | Controls change fewer words than the swap, so the baseline is weaker than intended |
| K distinct controls could be generated (K = `n_controls`, default 5) | `controls_short` | no | Δ averages over fewer controls, so it is noisier |

**Which control generator to use.** There are two (§3.3, `controls`):

- `synonyms` (the default, kept so the POC reproduces exactly) only changes words
  listed in `resources/synonyms.json`, which has 49 entries written for the 10 sample
  posts. On a new dataset almost every row is flagged `no_controls`, and the results
  carry a warning saying so.
- `masked-lm` works on arbitrary text: a small helper masked language model
  (`distilroberta-base` by default) proposes in-context replacements for content
  words, and a sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
  drops any control whose similarity to the original falls below `min_similarity`
  (default 0.85). **Use this for any real dataset.** A word can be changed if it is an
  ASCII word of at least 3 letters that is not a stopword, a gender word or a protected
  toxic word; a replacement must itself be a whole word that is none of those. Words
  listed in `synonyms.json` keep using their listed synonyms unless
  `use_synonym_list: false`.

Further guidance that the code does not check yet:

- **One referent per text.** A text about a man and a woman ("he told his sister ...")
  gets both swapped, which is a different intervention. Keep such rows out.
- **Avoid swap traps** that a word-level lexicon handles badly: the interjection
  "Man, ...", generic "you guys", slang address terms with no clean counterpart (bro,
  dude, sis), and quoted speech about a third person.
- **Names.** Only the 10 name pairs in the lexicon are swapped (James/Mary,
  John/Jane, ...). Other names are left unchanged and are not counted when detecting
  the text's gender, which can produce an inconsistent text ("Priya said he ...").
  Use lexicon names or none.
- **What counts as a word.** A word is a run of ASCII letters, so "he's" is "he" + "s"
  (and becomes "she's"), and accented names such as "José" are split and not matched.
- **Length.** Texts longer than a model's input limit are truncated (512 tokens for
  BERT-family models). If a gendered word sits after the cut, the model sees no
  difference. Keep texts well under the limit of the smallest model in the run.
- **English only**, matching the lexicons.

### 2.5 Dataset composition

- **Balance the groups.** Use equal numbers of rows whose original text refers to each
  group. If most originals were male-referencing, "female version" would coincide with
  "perturbed version" and `direction` could not separate the two. The results warn
  when the eligible rows are unbalanced.
- **Cover the uncertain middle.** Bias shows up where a model is unsure. A dataset of
  only obviously clean or obviously toxic texts pushes every probability to 0 or 1, all
  JSDs towards 0, and Δ towards an uninformative 0. Include mild and borderline
  content.
- **Size.** The per-model test is a one-sided paired t-test on Δ, Bonferroni-corrected
  across the models in the run. The table gives the number of **eligible** rows (after
  the exclusions in §2.4) needed for 80% power, by standardised effect size
  `mean(Δ) / sd(Δ)` (reported per model as `effect_size`):

  | Effect size | 1 model (α = 0.05) | 2 models (α = 0.025 each) | 3 models (α ≈ 0.017 each) |
  |-|-|-|-|
  | 0.5 (large) | 27 | 34 | 38 |
  | 0.3 (moderate) | 71 | 90 | 101 |
  | 0.2 (small) | 156 | 199 | 223 |

  With the 10 sample posts and 2 models, only effects of about 1.0 standard deviation
  are detectable, which is why the sample run's verdict is "insufficient evidence".
  For a real study, aim for **at least 200 eligible rows, balanced by group**.
- **A ready-made validation set.** `scripts/build_validation_set.py` builds 200
  gender-balanced rows (100 per group) from the public TweetEval `offensive` subset,
  with its offensive / not-offensive annotation as the gold label. It needs network
  access to the Hugging Face Hub (Colab has it); see `configs/validation.yaml`.

### 2.6 Privacy and licensing for real posts

The framework does not collect data. If you bring real social media posts:

- Check that the platform's terms and the dataset's licence allow research use and
  storage of the text. The validation script downloads TweetEval at run time and
  writes it under `data/validation/`, which is not committed.
- Replace user handles and URLs with placeholders (`@user`, `http://url`, the
  convention TweetEval already uses).
- Do not include names of private individuals. Because names are swap targets, replace
  them with lexicon names rather than deleting them.
- The results JSON contains every original and perturbed text. Treat results files
  with the same care as the dataset.

## 3. Models, task and run configuration

### 3.1 Task: 2 to 4 toxicity levels

Every model in a run is scored over the same ordered set of toxicity levels, least to
most toxic. A run uses one task:

| Task | Levels | Counted as toxic (`toxic_from`) |
|-|-|-|
| `toxicity` (default) | non-toxic, toxic | toxic |
| `toxicity-3` | non-toxic, borderline, toxic | toxic |
| `toxicity-4` | non-toxic, mild, toxic, severe | toxic and severe |

A custom task is a block with 2 to 4 unique `levels` and an optional `toxic_from`
(default: the second level) and `instruction` (read only by prompted models, planned).

How the levels are used:

- **JSD** (and therefore Δ) compares full distributions over all levels.
- **Flips** compare the most probable level (a tie goes to the more toxic level; for 2
  levels this is the POC's P(toxic) ≥ 0.5 rule).
- **Direction** is measured on the **severity score**, the expected level scaled to
  [0, 1]: with K levels, `severity = sum over levels of (level index × probability) / (K − 1)`.
  For 2 levels it equals P(toxic), exactly as in the POC.
- **`p_toxic_*`** fields in the results are the probability of the levels from
  `toxic_from` upwards, for readability.

### 3.2 How a classifier's labels map onto the levels

Fine-tuned classifiers have their own label names (`toxic`, `neutral`, `LABEL_1`,
`offensive`, ...). Each model entry says how they map onto the task's levels. Matching
is case-insensitive.

| Model head | 2-level task | 3 or 4-level task |
|-|-|-|
| **Multi-label** (independent sigmoid per head, e.g. `unitary/toxic-bert`) | `toxic_label` (default `toxic`) names the head read as P(toxic); other heads are ignored | Not possible: separate sigmoids do not form one distribution over levels. Use a 2-level task for such models |
| **Softmax** (one distribution over labels) | `toxic_label` feeds the top level and every other label the bottom one, or give a `label_map` | Give a `label_map`, or use a model whose labels are exactly the level names |

A `label_map` lists, per level, the model labels whose probabilities are summed into
it. The lowest level may be left out, in which case it receives every label not mapped
elsewhere:

```yaml
label_map: {non-toxic: [normal], borderline: [offensive], toxic: [hate speech]}
```

Every model label must end up in exactly one level and every level must get at least
one label; otherwise the model fails with a message listing the model's labels and the
task's levels. The mapping actually used is written to the results
(`run.models[].label_mapping`).

### 3.3 Run config file

`biascompare run CONFIG` reads a `.json`, `.yaml` or `.yml` file. Relative paths are
resolved against the config file's folder. Unknown keys are errors that list the
allowed keys. Examples: `configs/poc.yaml`, `configs/validation.yaml`.

**Run-level fields**

| Field | Default | Meaning |
|-|-|-|
| `dataset` | required | Path to the dataset file |
| `models` | required | Two or more model entries; the first one scored successfully is the reference |
| `task` | `toxicity` | A built-in task name, or a task block (§3.1) |
| `axis` | `gender` | Which lexicon and attribute to test (only `gender` exists) |
| `controls` | `synonyms` | `synonyms`, `masked-lm`, or a controls block (below) |
| `n_controls` | 5 | K, control perturbations per row |
| `seed` | 0 | Seeds control sampling and bootstraps |
| `alpha` | 0.05 | Significance level, before Bonferroni |
| `device` | `auto` | `auto` (cuda, then mps, then cpu), or a device name; also used by the masked-LM helpers unless they set their own |
| `output` | `results/<config name>.json` next to the config | Where `biascompare run` writes the results |

**Controls block**

| Field | Default | Meaning |
|-|-|-|
| `method` | `synonyms` | `synonyms` or `masked-lm` |
| `mlm_model` | `distilroberta-base` | Helper masked LM proposing replacements |
| `similarity_model` | `sentence-transformers/all-MiniLM-L6-v2` | Helper model for the similarity filter |
| `top_k` | 10 | Replacement candidates kept per word |
| `min_similarity` | 0.85 | Controls below this cosine similarity to the original are dropped. Not yet calibrated on real data; check the `similarity` values in the results |
| `use_synonym_list` | `true` | Words listed in `synonyms.json` keep their listed synonyms |
| `device` | `auto` | Where the helper models run |

**Model entry (`type: hf-classifier`, the only type in Phase 1)**

| Field | Required | Meaning |
|-|-|-|
| `type` | no | `hf-classifier` (default) |
| `name` | yes | Hugging Face Hub repo id, or a local folder |
| `toxic_label` | no | Shorthand for 2-level tasks (§3.2), default `toxic` |
| `label_map` | no | Model labels per level (§3.2) |
| `revision` | no | Hub commit hash to pin; the resolved revision is recorded |
| `token_env` | no | Name of an environment variable holding an access token |
| `batch_size`, `max_length` | no | Defaults 16 and the model's limit capped at 512 |

**Planned model types** (Phases 2 to 4): `hf-causal-lm` (local LLMs: `dtype`,
`quantization`, `use_chat_template`, `system`, `scoring`), `hf-masked-lm` (prompted
BERT: `template`, `verbalizers`) and `api-llm` (hosted LLMs: `provider`, `model`,
`estimation`, `n_samples`, `api_key_env`). See PLAN.md §4.

The same run in Python:

```python
from biascompare import ModelSpec, compare_models, load_config, run

results = run(load_config("configs/poc.yaml"))
# or
results = compare_models(
    [ModelSpec("unitary/toxic-bert"),
     ModelSpec("s-nlp/roberta_toxicity_classifier", label_map={"toxic": ["toxic"], "non-toxic": ["neutral"]})],
    "data/posts.jsonl", task="toxicity", controls="masked-lm",
)
```

### 3.4 Secrets

Tokens and API keys never go in a dataset or config file; a config containing a
`token` key is rejected. Name an environment variable with `token_env`, set `HF_TOKEN`,
or pass `token=` to `ModelSpec` in code. Tokens are excluded from `repr()`, logs and
results.

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
- `direction` is the second group's version minus the first group's.
- Planned: one such file per axis, registered in `biascompare/axes.py`.

### 4.2 `synonyms.json`

```json
{"dramatic": ["wild", "stunning", "intense"], "helpful": ["useful", "handy"]}
```

Lowercase words and single-word replacements that do not change meaning or toxicity.
The only source of controls under `synonyms`, and an override under `masked-lm`.

### 4.3 `protected_words.json`

```json
{"words": ["stupid", "idiot", "useless", "hate"]}
```

Words that controls must never change or introduce, because that would change the
label (swapping "stupid" for "smart" makes a post less toxic and would inflate the
baseline, hiding bias). Add the toxic and insulting vocabulary of your dataset here.

### 4.4 `stopwords.json`

Function words (`the`, `and`, `is`, contraction pieces like `don`, ...) that masked-LM
controls neither change nor introduce, so controls change content words, as the
synonym list does.

## 5. Output

`save_results(results, path)` (or `biascompare run`) writes one JSON file, schema
version `0.2`. The v0.1 fields are described in [`DESIGN.md` §7](../DESIGN.md#7-output-resultsjson)
and are all still present with the same meaning. Phase 1 adds:

- `run.task` (name, levels, `toxic_from`, instruction) and `run.controls` (method and
  settings).
- `run.models[]`: `type`, `revision`, and for classifiers the `label_mapping`, `head`
  and `max_length` actually used.
- `posts[]`: `attributes`, `label`, `meta`; each control's `similarity` (masked-LM
  controls only, otherwise `null`).
- `models[].per_post[]`: the full distributions `p_original`, `p_gender_swap`,
  `p_controls` (ordered as `run.task.levels`) and `predicted_original`.
- `models[].summary`: `effect_size`, `n_eligible`, `accuracy` and `n_labeled` (accuracy
  is `null` when no row has a gold label).
- `warnings`: plain-English data-quality warnings (excluded rows, unbalanced groups,
  low row count).

Undefined statistics (too few rows, zero variance) are written as `null`, never as
`NaN`. Later changes are listed in [`PLAN.md` §8](./PLAN.md#8-results-schema).

## 6. Checklist before a real run

1. `biascompare preview CONFIG` runs and shows sensible gender swaps.
2. Controls are `masked-lm`, and the `similarity` values of a sample of controls look
   like meaning-preserving rewordings.
3. Fewer than 5% of rows carry an excluding flag (`no_gender_terms`,
   `gender_mismatch`, `no_controls`).
4. The eligible rows are balanced between the two groups (no warning about it).
5. The eligible row count meets the size table in §2.5 for the effect you care about.
6. Texts are within the input limit of every model in the run.
7. No secrets in the dataset or config, and handles, URLs and private names replaced.
