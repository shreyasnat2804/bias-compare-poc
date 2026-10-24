"""Figures and tabular views built from a `compare_models` results dict.

`matplotlib` is imported lazily, inside `plot_comparison` only, so importing
this module — and building `summary_rows`, which needs no plotting library
at all — never requires matplotlib to be installed. This is what lets the
rest of the package (and its tests) run in an environment where matplotlib
is absent, per DESIGN.md §8's "heavy imports are lazy" convention.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from biascompare.perturbation_plan import is_eligible_flags

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_BAR_METRICS = ("sensitivity", "raw_swap_effect", "mean_delta")


def _bar_metric_labels(axis: str) -> tuple[str, str, str]:
    return (
        "Wording sensitivity\n(mean control JSD)",
        f"Raw {axis} effect\n(mean swap JSD)",
        f"Mean Δ\n({axis} effect beyond sensitivity)",
    )


def _axis_name(results: dict) -> str:
    return results.get("run", {}).get("axis", "gender")


def _group_labels(results: dict) -> tuple[str, str]:
    """(target label, other label) for the run's axis, e.g. ("women", "men")."""
    from biascompare.axes import BUILTIN_AXES, AxisSpec

    name = _axis_name(results)
    for public in results.get("run", {}).get("axes", []):
        if public["name"] == name:
            spec = AxisSpec(
                name=public["name"], groups=tuple(public["groups"]), target_group=public["target_group"],
                group_labels=public.get("group_labels"),
            )
            return spec.label(spec.target_group), spec.label(spec.other_group)
    spec = BUILTIN_AXES.get(name, BUILTIN_AXES["gender"])
    return spec.label(spec.target_group), spec.label(spec.other_group)


def _model_id(model: dict) -> str:
    return model.get("id") or model["name"]


def plot_comparison(results: dict) -> "Figure":
    """Three-panel figure summarising a `compare_models` results dict.

    (a) grouped bars per model: wording sensitivity vs. raw swap effect
        vs. mean Δ (the axis effect beyond wording sensitivity);
    (b) per-post Δ for each model, with a thin line pairing each post's
        values across models so paired comparisons are easy to read;
    (c) the same pairing for per-post direction, with a zero line.

    Models that failed to score (`summary` is `None`) are skipped in every
    panel and named in the figure title instead. Raises `ValueError` if no
    model was successfully scored (nothing to plot). Panels (b) and (c) plot
    only posts eligible for statistics — the same posts the `summary`
    figures in panel (a) are computed from (DESIGN §5) — and silently skip
    any remaining non-finite value, so a post with zero usable control
    perturbations (flagged `no_controls`) or any other excluded post never
    appears in the per-post panels.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    successful = [model for model in results["models"] if model.get("summary") is not None]
    if not successful:
        raise ValueError("plot_comparison: no successfully scored models in results to plot")

    eligible_ids = _eligible_post_ids(results)
    colors = _build_colors(colormaps["tab10"], successful)
    axis = _axis_name(results)
    target, other = _group_labels(results)
    fig, (ax_bar, ax_delta, ax_direction) = plt.subplots(3, 1, figsize=(10, 13))

    _bar_panel(ax_bar, successful, colors, axis)
    _slope_plot(
        ax_delta,
        successful,
        "delta",
        colors,
        eligible_ids,
        title=f"Per-post Δ by model ({axis} swap JSD minus mean control JSD)",
        ylabel="Δ (bits)",
    )
    _slope_plot(
        ax_direction,
        successful,
        "direction",
        colors,
        eligible_ids,
        title=f"Per-post direction by model (toxicity score, {target} minus {other} version)",
        ylabel="Direction",
    )
    ax_direction.axhline(0.0, color="black", linewidth=0.8, linestyle="--")

    fig.suptitle(_figure_title(results))
    fig.tight_layout()
    return fig


def summary_rows(results: dict) -> list[dict]:
    """One flat dict per model summary, suitable for `pandas.DataFrame`.

    A model that failed to score gets a row with its `name` and `error`
    only; `direction_ci` is split into `direction_ci_low`/`direction_ci_high`
    and `prediction_shares` into one `share_<level>` column per level, so
    every value in the row is a scalar.
    """
    rows: list[dict] = []
    for model in results["models"]:
        row = {"model": _model_id(model), "name": model["name"], "error": model.get("error")}
        summary = model.get("summary")
        if summary is not None:
            row.update(_flatten_summary(summary))
        rows.append(row)
    return rows


def ranking_rows(results: dict) -> list[dict]:
    """The axis's ranking (models by mean Δ, most bias beyond wording
    sensitivity first), one flat dict per model, for `pandas.DataFrame`."""
    return [dict(row) for row in results.get("ranking", [])]


def comparison_rows(results: dict) -> list[dict]:
    """One flat dict per model-vs-model comparison: the mean difference in
    Δ and in direction (model minus reference), their bootstrap CIs, raw and
    Holm-adjusted p-values, and the verdict."""
    rows = []
    for comparison in results.get("comparisons", []):
        row = {"model": comparison["model"], "reference": comparison["reference"]}
        for key, prefix in (("delta_diff", "delta"), ("direction_diff", "direction")):
            diff = comparison[key]
            ci = diff.get("ci") or [None, None]
            row.update(
                {
                    f"{prefix}_diff": diff["mean"],
                    f"{prefix}_ci_low": ci[0],
                    f"{prefix}_ci_high": ci[1],
                    f"{prefix}_p": diff["p"],
                    f"{prefix}_p_holm": diff.get("p_holm", diff["p"]),
                }
            )
        row["verdict"] = comparison["verdict"]
        rows.append(row)
    return rows


def _flatten_summary(summary: dict) -> dict:
    flat = {key: value for key, value in summary.items() if key not in ("direction_ci", "prediction_shares")}
    ci = summary.get("direction_ci") or [None, None]
    flat["direction_ci_low"] = ci[0]
    flat["direction_ci_high"] = ci[1]
    for level, share in (summary.get("prediction_shares") or {}).items():
        flat[f"share_{level}"] = share
    return flat


def _eligible_post_ids(results: dict) -> set[str]:
    """Post ids eligible for statistics (DESIGN §5), read from the results
    dict's `posts[].flags` — the same rule `compare.py` applies when
    building each model's `summary`, so the per-post panels never show a
    post the summary numbers already exclude."""
    return {post["id"] for post in results["posts"] if is_eligible_flags(post["flags"])}


def _figure_title(results: dict) -> str:
    title = f"{_axis_name(results).capitalize()}-bias comparison"
    failed = [_model_id(model) for model in results["models"] if model.get("summary") is None]
    if not failed:
        return title
    return f"{title} (failed to score: " + ", ".join(failed) + ")"


def _build_colors(cmap, successful: list[dict]) -> dict[str, tuple]:
    """A consistent colour per model id, reused across every panel."""
    return {_model_id(model): cmap(index % 10) for index, model in enumerate(successful)}


def _short_label(model: dict) -> str:
    if model.get("id") and model["id"] != model["name"]:
        return model["id"]
    return model["name"].rstrip("/").split("/")[-1]


def _num(value: float | None) -> float:
    return math.nan if value is None else float(value)


def _bar_panel(ax, successful: list[dict], colors: dict[str, tuple], axis: str) -> None:
    n_models = len(successful)
    bar_width = 0.8 / max(n_models, 1)
    x_base = list(range(len(_BAR_METRICS)))

    for index, model in enumerate(successful):
        offset = (index - (n_models - 1) / 2) * bar_width
        xs = [x + offset for x in x_base]
        ys = [_num(model["summary"][metric]) for metric in _BAR_METRICS]
        ax.bar(xs, ys, width=bar_width * 0.9, color=colors[_model_id(model)], label=_short_label(model))

    ax.set_xticks(x_base)
    ax.set_xticklabels(_bar_metric_labels(axis))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Jensen-Shannon divergence (bits)")
    ax.set_title(f"Wording sensitivity vs. raw {axis} effect vs. {axis} effect beyond sensitivity")
    ax.legend()


def _slope_plot(
    ax,
    successful: list[dict],
    key: str,
    colors: dict[str, tuple],
    eligible_ids: set[str],
    title: str,
    ylabel: str,
) -> None:
    """Plot `key` ("delta" or "direction") per eligible post, one column of
    points per model, with a thin line pairing each post's values across
    models. Posts outside `eligible_ids` are never plotted here — the same
    restriction `compute_model_summary` applies — and any individual value
    that is still non-finite (e.g. from a scorer misbehaving on one post) is
    dropped rather than plotted or connected."""
    model_labels = [_short_label(model) for model in successful]
    x_positions = list(range(len(successful)))

    eligible_rows = [
        [_num(model["per_post"][post_index][key]) for model in successful]
        for post_index, post in enumerate(successful[0]["per_post"])
        if post["post_id"] in eligible_ids
    ]

    for row in eligible_rows:
        if all(math.isfinite(value) for value in row):
            ax.plot(x_positions, row, color="0.75", linewidth=1.0, zorder=1)

    for model_index, model in enumerate(successful):
        ys = [row[model_index] for row in eligible_rows if math.isfinite(row[model_index])]
        ax.scatter(
            [x_positions[model_index]] * len(ys),
            ys,
            color=colors[_model_id(model)],
            label=model_labels[model_index],
            zorder=2,
            s=40,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
