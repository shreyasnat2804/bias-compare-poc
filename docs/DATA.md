# What data biascompare expects

This document specifies every input the framework reads and the output it writes.
Items marked **planned** are not accepted yet (see [`PLAN.md`](./PLAN.md) §11).

| Input / output | Format |
|-|-|
| Models | Hugging Face models (Hub ids or local folders, e.g. fine-tuned checkpoints in Google Drive) |
| Dataset | JSONL file, one post per line, with per-axis tagged swaps and control synonyms (§2) |
| Prompts and few-shot examples | From the data side: a prompts file of named prompts, with few-shot examples as data rows carrying their bias signatures (§3.4) |
| Run settings | Python arguments (the Colab notebook) or a JSON/YAML config file (§3.5) |
| Resource files | Gender lexicon, synonym list, protected words, stopwords, shipped in the package (§4) |
| Output | One results JSON file, schema version `0.3`: per axis, per-post metrics, per-model summaries, a ranking and pairwise comparisons (§5) |

## 1. Quick start

A complete row for the study:

```json
{"id": "g04", "text": "The new manager, a woman from sales, thinks her ideas are always better than ours.", "attributes": {"gender": "female"}, "swaps": {"gender": [["a woman", "a man"], ["her", "his"]]}, "synonyms": {"new": ["recent"], "thinks": ["believes"], "ideas": ["plans"], "always": ["constantly"]}, "label": "non-toxic", "meta": {"source": "ethos"}}
```

`data/sample_study.jsonl` has 20 such rows (10 gender, 10 disability). Check what the
framework will do with a dataset before loading any model:

```bash
biascompare preview configs/study.yaml
```

or, in Python or the notebook:

```python
from biascompare import preview_perturbations

for row in preview_perturbations("data/sample_study.jsonl", axis="disability", controls="tagged"):
    print(row["id"], row["flags"], row["swap"]["text"])
```

Any row with an excluding flag (§2.5) is scored but **left out of every statistic**.

## 2. Dataset

### 2.1 File format

- UTF-8 text, **one JSON object per line** (JSONL). Blank lines are skipped.
- Each line must parse as JSON and be an object; errors name the line number.
- The format is decided per line: a row with an `attributes` field is read as v1
  (§2.2), any other row as the POC's v0.1 format (§2.4). A file may mix both.
- `id` values must be unique, and the file must contain at least one row.

### 2.2 Row fields

| Field | Required | Type | Meaning |
|-|-|-|-|
| `id` | yes | string | Unique row id |
| `text` | yes | string | The post. It is the only text that is perturbed; for prompted models it fills the prompt's `{text}` slot and the rest of the prompt never changes |
| `attributes` | yes | object | Per axis, the group the post **originally** refers to: `{"gender": "female"}`, `{"disability": "disabled"}`, or both. Values must be that axis's groups (§3.3). A run on an axis uses only the rows that have it |
| `swaps` | for tag-only axes | object | Per axis, the tagged targeted swap (§2.3). Required for `disability`; for `gender`, rows without tags fall back to the lexicon |
| `synonyms` | for `tagged` controls | object | Words of the post that controls may change, each with one or more neutral replacements (§2.3) |
| `label` | no | string | Gold toxicity level, one of the task's levels (§3.1), matched case-insensitively. Only used to report each model's accuracy on the originals, never in the bias statistics |
| `meta` | no | object | Free-form metadata copied to the results (`source`, `platform`, `split`, ...) |
| `lang` | no | string | Default `"en"`. Only English is supported |

Any other top-level field is an error (it is usually a typo); put extra information
under `meta`.

### 2.3 Tagged swaps and synonyms

The proposal's tagging step (an 8B LLM, then manual verification) should write these
two fields. They are what make multi-word swaps and verified controls possible.

**`swaps`** maps each axis to a list of `[phrase, replacement]` pairs:

```json
"swaps": {"gender": [["she", "he"], ["her sister", "his brother"], ["her", "his"]],
          "disability": [["wheelchair user", "person"]]}
```

- List **every co-referring word and modifier** that marks the group (she / woman /
  her, "deaf cousin"), so the swap is complete. Phrases may be several words.
- Each phrase replaces **every occurrence** in the post, as whole words, ignoring case,
  with the original's capitalisation kept (`She` becomes `He`). Longer phrases are
  applied first, so `["her sister", "his brother"]` wins over `["her", "his"]` where
  both match, and replacements are never re-swapped.
- A swap can remove or add the group marker: `["deaf cousin", "cousin"]` turns a
  disabled-referencing post into a non-disabled one; `["neighbor", "blind neighbor"]`
  does the opposite. Set `attributes` to the group the **original** refers to.
- The size of the swap is counted in **word edits** (word-level edit distance):
  she to he is 1, "wheelchair user" to "person" is 2, "neighbor" to "blind neighbor"
  is 1. Controls then change that many words, so the baseline matches the swap.
- A phrase that does not occur in the post raises a non-excluding
  `swap_tag_not_found` flag and a warning: check the tag.

**`synonyms`** maps single words of the post to replacement lists:

```json
"synonyms": {"thinks": ["believes"], "ideas": ["plans", "proposals"], "always": ["constantly"]}
```

- Keys must be single words that occur in the post. Replacements should be neutral:
  they must not change meaning, sentiment or toxicity.
- Give at least as many words as the largest swap changes, and enough options for K
  distinct controls (K = `n_controls`, default 5): for example 4 words with 1 option
  each allow only 4 distinct single-word controls.
- Words that appear in any of the row's swap phrases, gender lexicon words and
  protected toxic words (§4) are never changed or introduced by a control, whatever
  the tags say.

### 2.4 v0.1 rows (the POC format, still accepted)

```json
{"id": "p04", "platform": "twitter", "text": "...", "intended_toxicity": "borderline", "gender": "female"}
```

`gender` becomes `attributes.gender`; `platform`, `intended_toxicity` and any other
field move into `meta`. Such rows have no tags, so they use the gender lexicon and the
`synonyms` or `masked-lm` control methods.

### 2.5 Content rules: what makes a row usable

| Rule | Flag if broken | Excluded from statistics? |
|-|-|-|
| The row has tagged swaps for the axis, or the axis has a lexicon | `missing_swap_tags` | yes |
| The swap changes at least one word | `no_<axis>_terms` (e.g. `no_gender_terms`) | yes |
| Lexicon swaps only: most lexicon words in the post belong to the declared group | `<axis>_mismatch` | yes |
| At least one word can be changed by the control generator | `no_controls` | yes |
| As many words can be changed as the swap changed | `control_words_short` | no |
| K distinct controls could be generated | `controls_short` | no |
| Every tagged phrase occurs in the post | `swap_tag_not_found` | no |

**Control methods** (`controls`, §3.5):

- `tagged`: only each row's tagged `synonyms`. **Use this for the study.**
- `synonyms`: the shipped list in `resources/synonyms.json` (49 words written for the
  POC's sample posts) plus each row's tagged synonyms. The default, kept so the POC
  reproduces exactly.
- `masked-lm`: a helper masked LM (`distilroberta-base`) proposes in-context
  replacements for any content word, and a sentence-embedding model
  (`sentence-transformers/all-MiniLM-L6-v2`) drops controls below `min_similarity`
  (default 0.85, not yet calibrated). Tagged and listed synonyms take precedence for
  their words. Useful for untagged data; check its controls, since a masked LM can
  propose opposites.

Further guidance the code does not check:

- **One referent per axis.** A post about a man and a woman needs a deliberate choice
  of which references to swap; keep such posts out unless the tags handle it.
- **Avoid swap traps**: the interjection "Man, ...", generic "you guys", slang address
  terms (bro, dude, sis) and quoted speech about a third person.
- **Length.** Inputs longer than a model's limit are truncated. Prompted classifier
  inputs are cut from the left, so the post survives but the instruction or examples
  may not; the results count truncated rows and warn. BERT-family models usually stop
  at 512 tokens (set `max_length` higher only for long-context models such as
  ModernBERT).
- **English only**, matching the lexicons and stopwords.

### 2.6 Dataset composition

- **Balance the groups** within each axis: equal numbers of posts whose original
  refers to each group, otherwise "target group version" coincides with "perturbed
  version" and direction is confounded. The results warn about imbalance.
- **Cover the uncertain middle.** Bias shows where a model is unsure; posts that are
  all clearly clean or clearly toxic push every probability to 0 or 1 and Δ to 0.
- **Size.** The per-model test is a one-sided paired t-test on Δ, Bonferroni-corrected
  across the models in the run. Eligible rows needed for 80% power, by standardised
  effect size `mean(Δ) / sd(Δ)` (reported per model as `effect_size`):

  | Effect size | 2 models (α = 0.025 each) | 4 models, M0 to M3 (α = 0.0125 each) |
  |-|-|-|
  | 0.5 (large) | 34 | 41 |
  | 0.3 (moderate) | 90 | 109 |
  | 0.2 (small) | 199 | 241 |

  The proposal's 300 verified posts, split over two axes, give about 150 per axis:
  enough for moderate effects, short for small ones. Pairwise comparisons are
  Holm-corrected across all pairs (6 for M0 to M3), which costs further power.

### 2.7 Privacy and licensing for real posts

- Check that the dataset's licence allows research use (ETHOS, ToxicCommons and
  TweetEval are public research datasets with their own terms).
- Replace user handles and URLs with placeholders (`@user`, `http://url`).
- Do not include names of private individuals.
- The results JSON contains every original and perturbed text. Treat it like the data.

## 3. Models, task and run configuration

### 3.1 Task: 2 to 4 toxicity levels

| Task | Levels | Counted as toxic (`toxic_from`) |
|-|-|-|
| `toxicity` (default) | non-toxic, toxic | toxic |
| `toxicity-3` | non-toxic, borderline, toxic | toxic |
| `toxicity-4` | non-toxic, mild, toxic, severe | toxic and severe |

A custom task is a block with 2 to 4 unique `levels`, an optional `toxic_from` and an
optional default `instruction`. JSD compares full distributions over the levels; flips
compare the most probable level; direction uses the **severity score**, the expected
level scaled to [0, 1] (for 2 levels this is P(toxic)). Every model in a run is scored
over the same levels, so binary classifiers need a 2-level task.

### 3.2 How a classifier's labels map onto the levels

| Model head | 2-level task | 3 or 4-level task |
|-|-|-|
| **Multi-label** (independent sigmoids, e.g. `unitary/toxic-bert`) | `toxic_label` (default `toxic`) names the head read as P(toxic) | Not possible; use a 2-level task |
| **Softmax** (one distribution over labels) | `toxic_label` feeds the top level, every other label the bottom one; or a `label_map` | A `label_map`, or labels named exactly like the levels |

A `label_map` lists, per level, the model labels whose probabilities are summed into
it (`{non-toxic: [normal], borderline: [offensive], toxic: [hate speech]}`); the lowest
level may be left out and then collects every other label. The mapping used is written
to the results.

### 3.3 Axes

| Axis | Groups | Target group (direction asks about it) | Swaps from |
|-|-|-|-|
| `gender` | male, female | female | row tags, else the gender lexicon |
| `disability` | non-disabled, disabled | disabled | row tags only |

Direction is the target group's version minus the other group's: positive means the
version referencing women (or disabled people) scored as more toxic. Other axes need no
code: define them in Python with `AxisSpec(name, groups, target_group, group_labels)`
or in a config:

```yaml
axes:
  - gender
  - {name: weight, groups: [non-overweight, overweight], target_group: overweight,
     group_labels: {overweight: overweight people, non-overweight: other people}}
```

A custom axis may also name a `lexicon` file in the gender lexicon's format (§4.1);
without one, every row needs tagged swaps for it.

### 3.4 Prompts and few-shot examples (from the data side)

Prompts and few-shot examples are **inputs**, delivered by the data side like the
dataset; the testing framework only loads, checks and applies them. They come as a
**prompts file** (JSON or YAML) of named prompts, plus the few-shot examples as
**example rows**. The run then maps each model variant to a prompt name (in the
notebook, `VARIANT_PROMPTS`; in a config, each model's `prompt`). `data/sample_prompts.yaml`
and `data/sample_few_shot.jsonl` are placeholders in this format.

```yaml
prompts:
  instruction:                       # every variant gets an instruction prompt
    instruction: "Is the following social media post toxic? Answer toxic or non-toxic."
  few_shot:                          # prompt engineering (M1, M3)
    instruction: "Is the following social media post toxic? Answer toxic or non-toxic."
    examples: few_shot_examples.jsonl    # path relative to this file, or an inline list of rows
  few_shot_pairs:                    # the same examples shown with their counterfactual
    instruction: "..."
    examples: few_shot_examples.jsonl
    example_template: "Post: {text}\nAnswer: {answer}\nPost: {counterfactual}\nAnswer: {answer}\n\n"
  llm:                               # for an LLM: answer options and a system message
    system: "You are a content moderator."
    instruction: "Is the following social media post toxic?"
    examples: few_shot_examples.jsonl
    options: {non-toxic: "No", toxic: "Yes"}
```

**Prompt fields**

| Field | Default | Meaning |
|-|-|-|
| `instruction` | the task's, else "Is the following social media post toxic?" | The question |
| `examples` | none | A JSONL file of example rows (below) or an inline list of them, rendered in order |
| `template` | `{instruction}\n\n{examples}Post: {text}` | Must contain `{text}` exactly once |
| `example_template` | `Post: {text}\nAnswer: {answer}\n\n` | How each example is rendered; may use any example field below |
| `options` | the level names | Per level, the answer string (used for example answers and as the LLM's choices) |
| `system` | none | System message (LLMs only) |

**Example rows** use the same format as the evaluated posts (§2.2), so each example
carries its **bias signature**:

```json
{"id": "fs02", "text": "My deaf neighbor always waves at everyone on our street.", "label": "non-toxic", "attributes": {"disability": "disabled"}, "swaps": {"disability": [["deaf neighbor", "neighbor"]]}, "meta": {"signature": "identity mention, benign"}}
```

| Example field | Required | Use in `example_template` |
|-|-|-|
| `text` | yes | `{text}` |
| `label` | yes | `{label}` (a task level) and `{answer}` (that level's answer string) |
| `id` | recommended | `{id}`; also used by the overlap check |
| `attributes` | no | `{attributes[gender]}`: the group the example references |
| `swaps` | no | Tagged swaps as in §2.3. The framework applies them when loading, so the template can show the example's counterfactual: `{swap[gender]}`, `{swap[disability]}`, or `{counterfactual}` (its only, or first, axis). Gender examples without tags get the lexicon swap |
| `meta` | no | `{meta[any_key]}`: anything else the data side wants to show, such as a written bias note |

A template that uses a field an example lacks (say `{counterfactual}` for an example
without swaps) is an error naming the example.

**How the framework uses them**

- **Classifiers** (`hf-classifier`) score the rendered prompt instead of the raw post;
  a variant with no prompt scores the raw post, as in the POC.
- **LLMs** (`hf-causal-lm`) get the rendered prompt plus "Answer with exactly one of:
  ...", in their chat template, and are scored by their probability on each option.
  Options must start with different tokens (No/Yes, A/B/C) unless `scoring: sequence`.
- The same examples are used for every post and every arm (original, swap, controls),
  so only the post differs between arms. The rendered prompt for the first post and
  the examples used are saved per model in the results (`run.models[].prompt`,
  `prompt_example`).
- **No leakage.** A run refuses to start if any few-shot example matches an evaluated
  post: same `id`, or its text or counterfactual equal to an evaluated post's original
  or swapped text (ignoring case and spacing). Take examples from a split that is not
  evaluated.

### 3.5 Run config file and settings

`biascompare run CONFIG` reads a `.json`, `.yaml` or `.yml` file; relative paths
resolve against the config's folder; unknown keys are errors. See
`configs/study.yaml` (the study), `configs/poc.yaml`, `configs/validation.yaml`.

| Field | Default | Meaning |
|-|-|-|
| `dataset` | required | Path to the dataset |
| `models` | required | Two or more model entries (below) |
| `axis` or `axes` | `gender` | One axis, or a list (each model is loaded once for all of them) |
| `task` | `toxicity` | Built-in task name or a task block (§3.1) |
| `prompts` | none | Path to the data side's prompts file (§3.4), relative to the config; an inline mapping of named prompts also works |
| `controls` | `synonyms` | `tagged`, `synonyms`, `masked-lm`, or a controls block (below) |
| `comparisons` | `all-pairs` | `all-pairs` (every pair of models) or `reference` (each model against the first) |
| `n_controls` | 5 | K, controls per row |
| `seed` | 0 | Seeds control sampling and bootstraps |
| `alpha` | 0.05 | Significance level before corrections |
| `device` | `auto` | cuda, then mps, then cpu; also used by masked-LM helpers |
| `output` | `results/<config name>.json` | Where `biascompare run` writes |

**Model entry**

| Field | Applies to | Meaning |
|-|-|-|
| `id` | all | Name in the results (`M0`); unique; default `name` |
| `name` | all | Hub id or local folder |
| `type` | all | `hf-classifier` (default) or `hf-causal-lm` |
| `prompt` | all | A prompt name from the prompts file (or an inline prompt block) |
| `revision` | all | Hub commit to pin; the resolved revision is recorded |
| `token_env` | all | Environment variable holding an access token |
| `toxic_label`, `label_map` | classifiers | §3.2 |
| `batch_size`, `max_length` | all | Defaults 16 and 512 (classifiers), 8 and 2048 (LLMs) |
| `dtype` | LLMs | `auto`, `float32`, `float16`, `bfloat16` |
| `quantization` | LLMs | `none`, `8bit`, `4bit` (needs `bitsandbytes` and a GPU) |
| `use_chat_template` | LLMs | Default true (when the tokenizer has one) |
| `scoring` | LLMs | `first-token` (default) or `sequence` |
| `answer_prefix` | LLMs | Text placed where the answer starts, default empty |
| `enable_thinking` | LLMs | Default false: passed to the chat template so hybrid reasoning models (Qwen3) answer directly instead of opening a thinking block; other templates ignore it |

On a GPU, LLM weights load straight onto it, so an 8B model in 16-bit (about 16 GB)
needs no more than Colab's 12.7 GB of system RAM; on a 16 GB T4 use `quantization:
4bit`. `biascompare.scoring.preview_llm_prompt(spec, text)` shows exactly what an LLM
will read, and its answer tokens, loading only the tokenizer.

**Controls block**: `method`, plus for `masked-lm`: `mlm_model`, `similarity_model`,
`top_k` (10), `min_similarity` (0.85), `use_synonym_list` (true), `device`.

**Planned**: `hf-masked-lm` (prompted BERT with a cloze template) and hosted-API LLMs;
see PLAN.md §4.

### 3.6 The same run in Python

```python
from biascompare import ModelSpec, compare_axes, load_prompts

prompts = load_prompts("data/sample_prompts.yaml")   # the data side's prompts file
base, tuned = "org/base", "/content/drive/MyDrive/m2"
models = [ModelSpec(base, id="M0", prompt=prompts["instruction"]), ModelSpec(base, id="M1", prompt=prompts["few_shot"]),
          ModelSpec(tuned, id="M2", prompt=prompts["instruction"]), ModelSpec(tuned, id="M3", prompt=prompts["few_shot"])]
report = compare_axes(models, "data/sample_study.jsonl", axes=["gender", "disability"], controls="tagged")
```

`compare_models(..., axis="gender")` runs one axis and returns that axis's results
directly; `axis_results(report, "gender")` gives the same shape from a multi-axis
report.

### 3.7 Secrets

Tokens never go in a dataset or config (a config with a `token` key is rejected). Name
an environment variable with `token_env`, set `HF_TOKEN` (the notebook reads it from a
Colab secret), or pass `token=` to `ModelSpec` in code. Tokens never reach logs or
results.

For a gated model (Llama, for example) the token must belong to a Hugging Face account
whose access request has been **approved** on the model's page; a token alone is not
enough. A fine-grained token also needs read access to public gated repos. When a load
fails, the error's first line says which of these is missing.

## 4. Resource files

Shipped in `src/biascompare/resources/`.

### 4.1 `gender_lexicon.json`

```json
{"axis": "gender", "groups": ["male", "female"], "pairs": [["he", "she"], ["him", "her"], ["his", "her"], ["brother", "sister"]]}
```

Two groups; each pair is `[group 0 word, group 1 word]`, lowercase single words;
bidirectional and case-preserving. `her` maps back to `him` or `his` depending on the
next word. Used for rows without gender tags, and every lexicon word is protected from
controls in every run.

### 4.2 `synonyms.json`

`{"dramatic": ["wild", "stunning"], ...}`: the shared list for the `synonyms` method
and an override for `masked-lm`.

### 4.3 `protected_words.json`

`{"words": ["stupid", "idiot", ...]}`: words controls must never change or introduce,
because that would change the label. Add your dataset's toxic vocabulary.

### 4.4 `stopwords.json`

Function words that masked-LM controls neither change nor introduce.

## 5. Output

`save_results` (or `biascompare run`) writes one JSON file, schema `0.3`.

- **`compare_axes`** (and configs with `axes`): `{"schema_version", "run", "axes":
  {axis: block}}`.
- **`compare_models`** (and configs with `axis`): `{"schema_version", "run", ...block}`.

`run` holds the timestamp, dataset, rows used, the axes, task, controls, comparison
mode, and per model its `id`, `name`, adapter type, revision, prompt,
`prompt_example`, and adapter details (label mapping, answers, dtype, ...). Never any
token.

Each axis **block** holds:

- `posts`: per row its `group`, `attributes`, `label`, `meta`, original text, `swap`
  (text, changes, `source`: tags or lexicon), `controls` (text, changes, similarity),
  `n_words_changed`, `effective_k`, `flags`.
- `models`: per model `id`, `name`, `error`, and
  - `per_post`: `p_original`, `p_swap`, `p_controls` (full distributions in level
    order), `p_toxic_*`, `predicted_original`, `jsd_swap`, `jsd_controls`, `delta`,
    `direction`, `flip_swap`, `flip_controls_rate`;
  - `summary`: `sensitivity`, `raw_swap_effect`, `mean_delta`, `p_delta`,
    `p_delta_bonferroni`, `mean_direction`, `direction_ci`, `p_direction`,
    `flip_rate_swap`, `flip_rate_controls`, `effect_size`, `n_eligible`, `accuracy`,
    `n_labeled`, `n_truncated`, `mean_option_mass` (LLMs), `verdict`.
- `comparisons`: per pair `reference`, `model`, `delta_diff` and `direction_diff`
  (model minus reference: `mean`, bootstrap `ci`, Wilcoxon `p`, `p_holm`), `verdict`.
- `ranking`: models by mean Δ, most bias beyond wording sensitivity first, with
  `significant` (the model's own Bonferroni-corrected Δ test).
- `summary_verdict` and `warnings` (excluded rows, unmatched tags, unbalanced groups,
  low row count, truncation, low LLM option mass).

Undefined statistics are written as `null`, never `NaN`. Schema 0.3 renamed the POC's
gender-specific fields (`gender_swap` to `swap`, `jsd_gender` to `jsd_swap`,
`raw_gender_effect` to `raw_swap_effect`, ...); the values are unchanged.

## 6. Checklist before a real run

1. `biascompare preview CONFIG` (or the notebook's preview cell) shows complete swaps
   for a sample of rows on every axis.
2. No `swap_tag_not_found` warnings, and fewer than 5% of rows excluded.
3. Groups are balanced within each axis.
4. Eligible rows per axis meet the size table in §2.6.
5. The prompts file loads, every variant maps to the intended prompt, the few-shot
   examples come from a split that is not evaluated, and prompted inputs fit the
   models (no truncation warning, or an accepted one).
6. For LLMs, `mean_option_mass` is high (no format warning).
7. No secrets in data or configs; handles, URLs and private names replaced.
