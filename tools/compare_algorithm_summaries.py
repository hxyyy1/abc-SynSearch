from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SummarySpec:
    name: str
    path: Path


DEFAULT_SUMMARIES = tuple(
    SummarySpec(name, Path(f".synsearch_work/trial-0/{method}-summary.csv"))
    for name, method in (
        ("AlphaSyn", "mcts"), ("HybridSyn", "hybrid"), ("SASyn", "sa"),
        ("UCB1", "ucb1"), ("UCB1Prefix", "ucb1-prefix"), ("LinUCB", "linucb"),
    )
)

DEFAULT_OUTPUT = Path(".compare_results/aggregate.csv")
DEFAULT_DETAILS_OUTPUT = Path(".compare_results/details.csv")

METRIC_COLUMNS = {
    "design": ("design_name", "file"),
    "and": ("final_and", "and"),
    "lev": ("final_lev", "lev"),
    "runtime": ("total_runtime_sec", "runtime_sec"),
    "memory": ("peak_memory_mb", "peak_memory_kb", "peak_memory", "memory_mb", "memory_kb", "memory"),
}


@dataclass(frozen=True)
class LoadedSummary:
    spec: SummarySpec
    design_column: str
    metric_columns: dict[str, str]
    rows: dict[str, dict[str, float | None]]
    variant_label: str | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare algorithm summaries using weighted per-design metric rank scores."
        )
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Algorithm summary to compare. Can be passed multiple times. "
            "Default: all six native algorithm summaries under .synsearch_work/trial-0."
        ),
    )
    parser.add_argument(
        "--missing-policy",
        choices=("common", "error"),
        default="common",
        help="Compare available results per design, or require identical design coverage.",
    )
    parser.add_argument("--and-weight", type=float, default=0.5)
    parser.add_argument("--lev-weight", type=float, default=0.2)
    parser.add_argument("--runtime-weight", type=float, default=0.2)
    parser.add_argument("--memory-weight", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV path for algorithm-level aggregate scores.",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=DEFAULT_DETAILS_OUTPUT,
        help="CSV path for per-design metric ranks and scores.",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Also print the comparison tables to stdout.",
    )
    return parser


def _parse_summary_specs(raw_specs: list[str]) -> list[SummarySpec]:
    if not raw_specs:
        return list(DEFAULT_SUMMARIES)

    specs: list[SummarySpec] = []
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise SystemExit(f"Invalid --summary '{raw_spec}'. Expected NAME=PATH.")
        name, raw_path = raw_spec.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise SystemExit(f"Invalid --summary '{raw_spec}'. Expected NAME=PATH.")
        specs.append(SummarySpec(name, Path(raw_path)))
    return specs


def _pick_column(headers: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    header_set = {header.strip() for header in headers}
    for candidate in candidates:
        if candidate in header_set:
            return candidate
    return None


def _parse_float(value: str, field_name: str, design_name: str, source: Path) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(
            f"Failed to parse '{field_name}'='{value}' for design '{design_name}' in {source}"
        ) from exc


def _is_missing_value(value: str) -> bool:
    return value.strip().lower() in {"", "-", "na", "n/a", "none", "null"}


def load_summaries(spec: SummarySpec) -> list[LoadedSummary]:
    path = spec.path.resolve()
    if not path.exists():
        raise SystemExit(f"Summary file not found for {spec.name}: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        design_column = _pick_column(headers, METRIC_COLUMNS["design"])
        if design_column is None:
            raise SystemExit(
                f"Could not find a design column in {path}. Expected one of: {METRIC_COLUMNS['design']}"
            )

        metric_columns: dict[str, str] = {}
        for metric_name in ("and", "lev", "runtime", "memory"):
            picked = _pick_column(headers, METRIC_COLUMNS[metric_name])
            if picked is not None:
                metric_columns[metric_name] = picked

        required = ("and", "lev", "runtime")
        missing_required = [metric for metric in required if metric not in metric_columns]
        if missing_required:
            raise SystemExit(
                f"Summary {path} is missing required columns for metrics: {', '.join(missing_required)}"
            )

        rows_by_variant: dict[str | None, dict[str, dict[str, float | None]]] = {}
        for raw_row in reader:
            design_name = (raw_row.get(design_column) or "").strip()
            if not design_name:
                continue
            variant_label = (raw_row.get("variant_label") or "").strip()
            variant_key = variant_label or None
            metrics: dict[str, float | None] = {}
            for metric_name, column_name in metric_columns.items():
                raw_value = (raw_row.get(column_name) or "").strip()
                if _is_missing_value(raw_value):
                    metrics[metric_name] = None
                    continue
                metrics[metric_name] = _parse_float(raw_value, column_name, design_name, path)
            variant_rows = rows_by_variant.setdefault(variant_key, {})
            if design_name in variant_rows:
                label_text = variant_label or "-"
                raise SystemExit(
                    f"Summary {path} contains duplicate rows for design '{design_name}' "
                    f"under variant_label '{label_text}'."
                )
            variant_rows[design_name] = metrics

    if not rows_by_variant:
        raise SystemExit(f"Summary {path} does not contain any usable rows.")
    return [
        LoadedSummary(
            spec=SummarySpec(spec.name, path),
            design_column=design_column,
            metric_columns=metric_columns,
            rows=rows,
            variant_label=variant_label,
        )
        for variant_label, rows in sorted(
            rows_by_variant.items(),
            key=lambda item: _natural_key(item[0] or ""),
        )
    ]


def _natural_key(text: str) -> tuple[tuple[int, object], ...]:
    parts: list[tuple[int, object]] = []
    current = ""
    in_digits = False
    for char in text:
        if char.isdigit():
            if not in_digits and current:
                parts.append((1, current.lower()))
                current = ""
            current += char
            in_digits = True
        else:
            if in_digits and current:
                parts.append((0, int(current)))
                current = ""
            current += char
            in_digits = False
    if current:
        parts.append((0, int(current)) if in_digits else (1, current.lower()))
    return tuple(parts)


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def _shared_designs(summaries: list[LoadedSummary], policy: str) -> list[str]:
    design_sets = [set(summary.rows) for summary in summaries]
    union = set.union(*design_sets)
    if not union:
        raise SystemExit("No designs were found across the provided summaries.")

    if policy == "error":
        shared = set.intersection(*design_sets)
        if shared != union:
            missing_lines: list[str] = []
            for summary in summaries:
                missing = sorted(union - set(summary.rows), key=_natural_key)
                if missing:
                    missing_lines.append(f"{summary.spec.name}: missing {', '.join(missing)}")
            detail = "\n".join(missing_lines)
            raise SystemExit(f"Summaries do not cover the same designs:\n{detail}")

    return sorted(union, key=_natural_key)


def _summary_has_metric(summary: LoadedSummary, design_name: str, metric_name: str) -> bool:
    if metric_name not in summary.metric_columns:
        return False
    value = summary.rows[design_name].get(metric_name)
    return isinstance(value, (int, float))


def _active_metrics(
    summaries: list[LoadedSummary],
    weights: dict[str, float],
) -> list[str]:
    active: list[str] = []
    for metric_name in ("and", "lev", "runtime", "memory"):
        if weights[metric_name] <= 0:
            continue
        if all(metric_name in summary.metric_columns for summary in summaries):
            active.append(metric_name)
    if not active:
        raise SystemExit("No metrics are available after applying the provided weights and summary columns.")
    return active


def _score_for_rank(rank: int) -> int:
    if rank <= 1:
        return 10
    if rank == 2:
        return 9
    if rank == 3:
        return 8
    if rank == 4:
        return 7
    if rank == 5:
        return 6
    return 5


def _compute_metric_ranks(
    eligible_summaries: list[LoadedSummary],
    design_name: str,
    metric_name: str,
) -> dict[tuple[str, str | None, str], tuple[int, int]]:
    ranked_values = sorted(
        [
            (
                float(summary.rows[design_name][metric_name]),
                summary.spec.name,
                summary.variant_label,
                str(summary.spec.path),
            )
            for summary in eligible_summaries
        ],
        key=lambda item: (
            item[0],
            _natural_key(item[1]),
            _natural_key(item[2] or ""),
        ),
    )
    metric_ranks: dict[tuple[str, str | None, str], tuple[int, int]] = {}
    previous_value: float | None = None
    current_rank = 0
    for index, (value, algorithm, variant_label, summary_path) in enumerate(ranked_values, start=1):
        if previous_value is None or value != previous_value:
            current_rank = index
            previous_value = value
        metric_ranks[(algorithm, variant_label, summary_path)] = (
            current_rank,
            _score_for_rank(current_rank),
        )
    return metric_ranks


def compare_summaries(
    summaries: list[LoadedSummary],
    *,
    weights: dict[str, float],
    missing_policy: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str], list[str], float]:
    designs = _shared_designs(summaries, missing_policy)
    metrics = _active_metrics(summaries, weights)
    active_weight_sum = sum(weights[metric_name] for metric_name in metrics)
    if active_weight_sum <= 0:
        raise SystemExit("The sum of active metric weights must be positive.")

    compared_designs: set[str] = set()
    detail_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    scores_by_summary: dict[tuple[str, str | None, str], list[dict[str, float | None]]] = {
        (summary.spec.name, summary.variant_label, str(summary.spec.path)): []
        for summary in summaries
    }

    for design_name in designs:
        summaries_for_design = [summary for summary in summaries if design_name in summary.rows]
        eligible_summaries = [
            summary
            for summary in summaries_for_design
            if all(_summary_has_metric(summary, design_name, metric_name) for metric_name in metrics)
        ]
        if len(eligible_summaries) < 2:
            continue
        design_metrics = list(metrics)
        if not design_metrics:
            continue
        compared_designs.add(design_name)
        design_weight_sum = sum(weights[metric_name] for metric_name in design_metrics)
        metric_rankings = {
            metric_name: _compute_metric_ranks(eligible_summaries, design_name, metric_name)
            for metric_name in design_metrics
        }
        for summary in eligible_summaries:
            raw_metrics = summary.rows[design_name]
            summary_key = (summary.spec.name, summary.variant_label, str(summary.spec.path))
            metric_scores = {
                metric_name: metric_rankings[metric_name][summary_key][1]
                for metric_name in design_metrics
            }
            metric_ranks = {
                metric_name: metric_rankings[metric_name][summary_key][0]
                for metric_name in design_metrics
            }
            final_score = math.fsum(
                weights[metric_name] * metric_scores[metric_name]
                for metric_name in design_metrics
            )
            scores_by_summary[summary_key].append(
                {
                    "final_score": final_score,
                    "weight_sum": design_weight_sum,
                    **{
                        f"raw_{metric_name}": float(raw_metrics[metric_name])
                        if metric_name in design_metrics
                        else None
                        for metric_name in metrics
                    },
                    **{
                        f"{metric_name}_rank": metric_ranks.get(metric_name)
                        for metric_name in metrics
                    },
                    **{
                        f"{metric_name}_score": metric_scores.get(metric_name)
                        for metric_name in metrics
                    },
                }
            )
            detail_row: dict[str, object] = {
                "algorithm": summary.spec.name,
                "variant_label": summary.variant_label,
                "design": design_name,
                "final_score": final_score,
                "active_weight_sum": design_weight_sum,
            }
            for metric_name in metrics:
                detail_row[f"{metric_name}"] = raw_metrics.get(metric_name)
                detail_row[f"{metric_name}_rank"] = metric_ranks.get(metric_name)
                detail_row[f"{metric_name}_score"] = metric_scores.get(metric_name)
            detail_rows.append(detail_row)

    if not detail_rows:
        raise SystemExit("No comparable designs were found across the provided summaries.")

    for summary in summaries:
        summary_key = (summary.spec.name, summary.variant_label, str(summary.spec.path))
        algorithm_scores = scores_by_summary[summary_key]
        if not algorithm_scores:
            continue
        row: dict[str, object] = {
            "algorithm": summary.spec.name,
            "variant_label": summary.variant_label,
            "design_count": len(algorithm_scores),
            "final_score": math.fsum(float(item["final_score"]) for item in algorithm_scores),
            "summary_path": str(summary.spec.path),
        }
        for metric_name in metrics:
            raw_values = [item[f"raw_{metric_name}"] for item in algorithm_scores if item[f"raw_{metric_name}"] is not None]
            rank_values = [item[f"{metric_name}_rank"] for item in algorithm_scores if item[f"{metric_name}_rank"] is not None]
            score_values = [item[f"{metric_name}_score"] for item in algorithm_scores if item[f"{metric_name}_score"] is not None]
            row[f"avg_{metric_name}"] = (sum(raw_values) / len(raw_values)) if raw_values else None
            row[f"avg_{metric_name}_rank"] = (sum(rank_values) / len(rank_values)) if rank_values else None
            row[f"avg_{metric_name}_score"] = (sum(score_values) / len(score_values)) if score_values else None
        aggregate_rows.append(row)

    aggregate_rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            _natural_key(str(row["algorithm"])),
            _natural_key(str(row.get("variant_label") or "")),
        )
    )
    ranked_detail_rows: list[dict[str, object]] = []
    for design_name in sorted(compared_designs, key=_natural_key):
        design_rows = [row for row in detail_rows if row["design"] == design_name]
        design_rows.sort(
            key=lambda row: (
                -float(row["final_score"]),
                _natural_key(str(row["algorithm"])),
                _natural_key(str(row.get("variant_label") or "")),
            )
        )
        previous_score: float | None = None
        current_rank = 0
        for index, row in enumerate(design_rows, start=1):
            score = float(row["final_score"])
            if previous_score is None or score != previous_score:
                current_rank = index
                previous_score = score
            ranked_row = {
                "design": row["design"],
                "rank": current_rank,
                "algorithm": row["algorithm"],
                "variant_label": row["variant_label"],
                "final_score": row["final_score"],
                "active_weight_sum": row["active_weight_sum"],
            }
            for metric_name in metrics:
                ranked_row[metric_name] = row[metric_name]
                ranked_row[f"{metric_name}_rank"] = row[f"{metric_name}_rank"]
                ranked_row[f"{metric_name}_score"] = row[f"{metric_name}_score"]
            ranked_detail_rows.append(ranked_row)
    detail_rows = ranked_detail_rows
    return aggregate_rows, detail_rows, sorted(compared_designs, key=_natural_key), metrics, active_weight_sum


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_optional_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _print_report(
    aggregate_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    *,
    designs: list[str],
    metrics: list[str],
    active_weight_sum: float,
    requested_weights: dict[str, float],
) -> None:
    print(f"Compared designs: {len(designs)}")
    print(f"Active metrics: {', '.join(metrics)}")
    if active_weight_sum < sum(requested_weights.values()):
        print(
            "Note: some weighted metrics were unavailable in the summaries and were excluded "
            f"(active weight sum = {active_weight_sum:.3f})."
        )

    aggregate_headers = ["rank", "algorithm", "variant_label", "designs", "final_score"]
    for metric_name in metrics:
        aggregate_headers.append(f"avg_{metric_name}")
        aggregate_headers.append(f"avg_{metric_name}_rank")
        aggregate_headers.append(f"avg_{metric_name}_score")
    aggregate_rows_text: list[list[str]] = []
    for rank, row in enumerate(aggregate_rows, start=1):
        cells = [
            str(rank),
            str(row["algorithm"]),
            str(row.get("variant_label") or "-"),
            str(row["design_count"]),
            f"{float(row['final_score']):.6f}",
        ]
        for metric_name in metrics:
            cells.append(_format_optional_float(row[f"avg_{metric_name}"]))
            cells.append(_format_optional_float(row[f"avg_{metric_name}_rank"]))
            cells.append(_format_optional_float(row[f"avg_{metric_name}_score"]))
        aggregate_rows_text.append(cells)
    print()
    print(_format_table(aggregate_headers, aggregate_rows_text))

    detail_headers = ["design", "rank", "algorithm", "variant_label", "final_score", "active_weight_sum"]
    for metric_name in metrics:
        detail_headers.extend((metric_name, f"{metric_name}_rank", f"{metric_name}_score"))
    detail_rows_text: list[list[str]] = []
    for row in detail_rows:
        cells = [
            str(row["design"]),
            str(row["rank"]),
            str(row["algorithm"]),
            str(row.get("variant_label") or "-"),
            f"{float(row['final_score']):.6f}",
            f"{float(row['active_weight_sum']):.3f}",
        ]
        for metric_name in metrics:
            cells.append(_format_optional_float(row[metric_name]))
            cells.append(_format_optional_float(row[f"{metric_name}_rank"]))
            cells.append(_format_optional_float(row[f"{metric_name}_score"]))
        detail_rows_text.append(cells)
    print()
    print(_format_table(detail_headers, detail_rows_text))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    specs = _parse_summary_specs(args.summary)
    loaded = [summary for spec in specs for summary in load_summaries(spec)]
    weights = {
        "and": args.and_weight,
        "lev": args.lev_weight,
        "runtime": args.runtime_weight,
        "memory": args.memory_weight,
    }

    aggregate_rows, detail_rows, designs, metrics, active_weight_sum = compare_summaries(
        loaded,
        weights=weights,
        missing_policy=args.missing_policy,
    )
    _write_csv(args.output, aggregate_rows)
    _write_csv(args.details_output, detail_rows)
    if args.print_report:
        _print_report(
            aggregate_rows,
            detail_rows,
            designs=designs,
            metrics=metrics,
            active_weight_sum=active_weight_sum,
            requested_weights=weights,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
