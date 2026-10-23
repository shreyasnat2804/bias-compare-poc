"""Build a gender-balanced validation set from a public dataset (docs/PLAN.md §11, Phase 1).

Downloads TweetEval's `offensive` subset (Barbieri et al. 2020; tweets with
user handles already replaced by `@user`), keeps tweets that refer to
exactly one gender through the lexicon, and writes a balanced v1 JSONL file
with the offensive/not-offensive annotation as the gold `label`.

Needs the `datasets` package (preinstalled on Colab) and network access to
the Hugging Face Hub. Run from the repository root:

    python scripts/build_validation_set.py

then check and run it with the matching config:

    biascompare preview configs/validation.yaml
    biascompare run configs/validation.yaml

The output file is not committed (data/validation/ is in .gitignore): the
tweets stay under the dataset's own terms.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from biascompare.axes import load_swapper

PER_GROUP = 100
SEED = 0
OUTPUT = Path("data/validation/tweet_eval_offensive_gender.jsonl")
LABELS = {0: "non-toxic", 1: "toxic"}
MAX_CHARS = 280


def _single_group(swapper, text: str) -> str | None:
    """The one gender group `text` refers to, or None if it refers to none or both."""
    present = [group for group, count in swapper.group_counts(text).items() if count > 0]
    return present[0] if len(present) == 1 else None


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("This script needs the `datasets` package: pip install datasets", file=sys.stderr)
        return 1

    swapper = load_swapper("gender")
    by_group: dict[str, list[dict]] = {group: [] for group in swapper.groups}
    for split in ("train", "validation", "test"):
        for index, row in enumerate(load_dataset("cardiffnlp/tweet_eval", "offensive", split=split)):
            text = " ".join(row["text"].split())
            if len(text) > MAX_CHARS:
                continue
            group = _single_group(swapper, text)
            if group is None:
                continue
            by_group[group].append(
                {
                    "id": f"te-{split}-{index}",
                    "text": text,
                    "attributes": {"gender": group},
                    "label": LABELS[row["label"]],
                    "meta": {"platform": "twitter", "source": f"tweet_eval/offensive/{split}"},
                }
            )

    rng = random.Random(SEED)
    rows: list[dict] = []
    for group, candidates in by_group.items():
        if len(candidates) < PER_GROUP:
            print(f"only {len(candidates)} usable {group} tweets, wanted {PER_GROUP}", file=sys.stderr)
        rng.shuffle(candidates)
        rows.extend(candidates[:PER_GROUP])
    rng.shuffle(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    counts = {group: sum(row["attributes"]["gender"] == group for row in rows) for group in swapper.groups}
    print(f"wrote {len(rows)} rows to {OUTPUT} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
