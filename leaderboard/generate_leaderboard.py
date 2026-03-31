#!/usr/bin/env python3
"""
Generate a static leaderboard HTML page from experiment outputs under results/.

The script scans each subdirectory in the provided results directory. If a
subdirectory contains both an exp_config.json and a metrics/BenchmarkMetrics.csv
(case-insensitive) file, it is included on the leaderboard.
"""

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


METRIC_FILENAMES = [
    "metrics/BenchmarkMetrics.csv",
    "metrics/benchmarkMetrics.csv",
    "metrics/benchmarkmetrics.csv",
]

VERSIONED_ID_RE = re.compile(r"^(?P<base>.+)_v(?P<version>\d+)$")

REQUIRED_STRING_KEYS = [
    "page_title",
    "title_link_text",
    "title_link_url",
    "title_suffix",
    "meta_text",
    "stats_loading",
    "hero_text",
    "controls_hint",
    "download_csv_label",
    "filters_title",
    "filters_hint",
    "filter_exp_type_label",
    "filter_env_label",
    "filter_task_label",
    "filter_network_label",
    "filter_action_all",
    "filter_action_none",
    "isolate_label",
    "show_all_label",
    "deselect_label",
    "collapse_folds_label",
    "merge_folds_label",
    "filter_summary",
    "filter_empty",
    "footer_hint",
    "footer_separator",
    "generated_label",
    "logo_alt",
    "stats_pill",
    "unknown_env",
    "unknown_task",
    "unknown_network",
    "na_label",
    "folds_tooltip",
    "csv_filename",
    "slug_default",
    "slug_all",
    "sort_indicator_asc",
    "sort_indicator_desc",
    "type_labels",
    "table_headers",
    "metric_descriptions",
]

REQUIRED_TABLE_HEADERS = [
    "rank",
    "exp_id",
    "algorithm",
    "script",
    "alg_config",
    "env_seed",
    "torch_seed",
]

REQUIRED_TYPE_LABELS = ["normal", "open", "cond_open"]

COLLAPSE_KEY_FIELDS = [
    "exp_type",
    "env_config",
    "task_config",
    "network",
    "algorithm",
    "script",
    "alg_config",
]


def read_metrics(exp_dir: Path) -> Optional[Dict[str, str]]:
    """Return the metrics dict (first row) and header order if present."""
    metrics_path = None
    for candidate in METRIC_FILENAMES:
        path = exp_dir / candidate
        if path.exists():
            metrics_path = path
            break

    if not metrics_path:
        return None

    with metrics_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None
        first_row = rows[0]
        header_order = reader.fieldnames or list(first_row.keys())
        return {"data": first_row, "header": header_order}


def read_config(exp_dir: Path) -> Optional[Dict]:
    config_path = exp_dir / "exp_config.json"
    if not config_path.exists():
        return None
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def split_versioned_id(exp_id: str) -> Tuple[str, Optional[int]]:
    match = VERSIONED_ID_RE.match(exp_id)
    if not match:
        return exp_id, None
    return match.group("base"), int(match.group("version"))


def average_metrics(experiments: Sequence[Dict], anchor_metrics: Dict[str, str]) -> Dict[str, object]:
    metric_keys = set()
    for exp in experiments:
        metric_keys.update((exp.get("metrics") or {}).keys())

    averaged: Dict[str, object] = {}
    for key in metric_keys:
        values: List[float] = []
        for exp in experiments:
            value = (exp.get("metrics") or {}).get(key)
            if value is None or value == "":
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(parsed):
                continue
            values.append(parsed)
        if values:
            averaged[key] = sum(values) / len(values)
        elif key in anchor_metrics:
            averaged[key] = anchor_metrics[key]
        else:
            averaged[key] = ""
    return averaged


def merged_metric_order(experiments: Sequence[Dict]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for exp in experiments:
        metric_order = exp.get("metric_order") or []
        for metric in metric_order:
            if metric in seen:
                continue
            seen.add(metric)
            ordered.append(metric)
        for metric in (exp.get("metrics") or {}).keys():
            if metric in seen:
                continue
            seen.add(metric)
            ordered.append(metric)
    return ordered


def collapse_key(exp: Dict) -> Tuple[str, ...]:
    return tuple(str(exp.get(field) or "") for field in COLLAPSE_KEY_FIELDS)


def pick_anchor(group: List[Dict]) -> Dict:
    def sort_key(item: Dict) -> Tuple[int, int, str]:
        _, version = split_versioned_id(item["exp_id"])
        has_version = 1 if version is not None else 0
        version_value = version if version is not None else -1
        return (has_version, version_value, item["exp_id"])

    return sorted(group, key=sort_key)[0]


def display_id_for_group(group: List[Dict], anchor: Dict) -> str:
    bases = {split_versioned_id(exp["exp_id"])[0] for exp in group}
    if len(bases) == 1:
        return next(iter(bases))
    return anchor["exp_id"]


def merge_seed_fields(merged: Dict, group: List[Dict]) -> None:
    env_seeds = {exp.get("env_seed") for exp in group if exp.get("env_seed") not in (None, "")}
    torch_seeds = {exp.get("torch_seed") for exp in group if exp.get("torch_seed") not in (None, "")}

    if len(env_seeds) > 1:
        merged["env_seed"] = "varies"
    elif env_seeds and merged.get("env_seed") in (None, ""):
        merged["env_seed"] = next(iter(env_seeds))

    if len(torch_seeds) > 1:
        merged["torch_seed"] = "varies"
    elif torch_seeds and merged.get("torch_seed") in (None, ""):
        merged["torch_seed"] = next(iter(torch_seeds))


def collapse_repeated_experiments(experiments: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, ...], List[Dict]] = {}
    for exp in experiments:
        grouped.setdefault(collapse_key(exp), []).append(exp)

    collapsed: List[Dict] = []
    for key in sorted(grouped.keys()):
        group = sorted(grouped[key], key=lambda item: item["exp_id"])
        if len(group) == 1:
            single = dict(group[0])
            single["fold_count"] = 1
            collapsed.append(single)
            continue

        anchor = pick_anchor(group)
        merged = dict(anchor)
        merged["exp_id"] = display_id_for_group(group, anchor)
        merged["metrics"] = average_metrics(group, anchor.get("metrics") or {})
        merged["metric_order"] = merged_metric_order(group)
        merged["fold_count"] = len(group)
        merged["fold_members"] = [exp["exp_id"] for exp in group]
        merge_seed_fields(merged, group)
        collapsed.append(merged)

    return collapsed


def validate_strings(strings: Dict, strings_path: Path) -> None:
    missing = [key for key in REQUIRED_STRING_KEYS if key not in strings]
    errors = []
    if missing:
        errors.append(f"Missing keys: {', '.join(sorted(missing))}")

    if not isinstance(strings.get("type_labels"), dict):
        errors.append("type_labels must be a JSON object")
    else:
        missing_type_labels = [key for key in REQUIRED_TYPE_LABELS if key not in strings["type_labels"]]
        if missing_type_labels:
            errors.append(f"type_labels missing: {', '.join(sorted(missing_type_labels))}")

    if not isinstance(strings.get("table_headers"), dict):
        errors.append("table_headers must be a JSON object")
    else:
        missing_headers = [key for key in REQUIRED_TABLE_HEADERS if key not in strings["table_headers"]]
        if missing_headers:
            errors.append(f"table_headers missing: {', '.join(sorted(missing_headers))}")

    if not isinstance(strings.get("metric_descriptions"), dict):
        errors.append("metric_descriptions must be a JSON object")

    if errors:
        detail = "; ".join(errors)
        raise SystemExit(f"Invalid strings file at {strings_path}: {detail}")


def load_strings(strings_path: Path) -> Dict:
    if not strings_path.exists():
        raise SystemExit(f"Strings file not found: {strings_path}")
    try:
        with strings_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Strings file is not valid JSON: {strings_path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Strings file must contain a JSON object: {strings_path}")
    validate_strings(payload, strings_path)
    return payload


def load_template(template_path: Path) -> str:
    if not template_path.exists():
        raise SystemExit(f"Template file not found: {template_path}")
    try:
        return template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Unable to read template file: {template_path}") from exc


def collect_experiments(results_dir: Path) -> List[Dict]:
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    experiments: List[Dict] = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name == "sample_results":
            continue

        config = read_config(exp_dir)
        metrics = read_metrics(exp_dir)
        if not config or not metrics:
            continue

        script_path = config.get("script") or ""
        algorithm = (
            config.get("algorithm")
            or config.get("baseline_model")
            or config.get("policy")
            or ""
        )

        experiments.append(
            {
                "exp_id": exp_dir.name,
                "exp_path": str(exp_dir.as_posix()),
                "exp_type": config.get("exp_type", "normal"),
                "env_config": config.get("env_config"),
                "task_config": config.get("task_config"),
                "network": config.get("network"),
                "algorithm": algorithm,
                "script": Path(script_path).name if script_path else "",
                "alg_config": config.get("alg_config")
                or config.get("algorithm_config")
                or config.get("algorithm_configuration")
                or "",
                "env_seed": config.get("env_seed"),
                "torch_seed": config.get("torch_seed"),
                "metrics": metrics["data"],
                "metric_order": metrics["header"],
            }
        )
    return experiments


def build_html(payload: Dict, output_path: Path, template: str) -> None:
    """Write a self-contained HTML file with embedded data and styling."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, indent=2)
    generated_at = payload["generated_at"]
    strings = payload["strings"]

    replacements = {
        "__PAGE_TITLE__": strings["page_title"],
        "__TITLE_LINK_URL__": strings["title_link_url"],
        "__TITLE_LINK_TEXT__": strings["title_link_text"],
        "__TITLE_SUFFIX__": strings["title_suffix"],
        "__META_TEXT__": strings["meta_text"],
        "__STATS_LOADING__": strings["stats_loading"],
        "__HERO_TEXT__": strings["hero_text"],
        "__CONTROLS_HINT__": strings["controls_hint"],
        "__DOWNLOAD_LABEL__": strings["download_csv_label"],
        "__FILTERS_TITLE__": strings["filters_title"],
        "__FILTERS_HINT__": strings["filters_hint"],
        "__ISOLATE_LABEL__": strings["isolate_label"],
        "__SHOW_ALL_LABEL__": strings["show_all_label"],
        "__DESELECT_LABEL__": strings["deselect_label"],
        "__COLLAPSE_FOLDS_LABEL__": strings["collapse_folds_label"],
        "__MERGE_FOLDS_LABEL__": strings["merge_folds_label"],
        "__FOOTER_HINT__": strings["footer_hint"],
        "__FOOTER_SEPARATOR__": strings["footer_separator"],
        "__GENERATED_LABEL__": strings["generated_label"],
        "__LOGO_ALT__": strings["logo_alt"],
        "__GENERATED_AT__": generated_at,
        "__DATA__": data_json,
    }

    html = template
    for token, value in replacements.items():
        html = html.replace(token, str(value))
    output_path.write_text(html, encoding="utf-8")


def build_experiment_links(
    experiments: List[Dict],
    repo_url: str,
    local_link_prefix: str,
) -> None:
    base_url = repo_url.rstrip("/") + "/" if repo_url else ""
    local_prefix = local_link_prefix.strip()

    for exp in experiments:
        if base_url:
            exp["exp_link"] = base_url + exp["exp_path"]
            continue

        relative = exp["exp_path"].lstrip("/")
        if local_prefix:
            prefix = local_prefix.rstrip("/")
            exp["exp_link"] = f"{prefix}/{relative}"
        else:
            exp["exp_link"] = relative


def infer_default_repo_url(strings: Dict) -> str:
    title_link_url = (strings.get("title_link_url") or "").strip()
    if not title_link_url or "github.com" not in title_link_url:
        return ""

    normalized = title_link_url.rstrip("/")
    if "/tree/" in normalized or "/blob/" in normalized:
        return normalized
    return f"{normalized}/tree/main"


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a static leaderboard page.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base directory containing experiment result subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/leaderboard"),
        help="Directory where the static site will be written.",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default="",
        help="Optional base URL for experiment links (e.g., https://github.com/org/repo/tree/main/).",
    )
    parser.add_argument(
        "--local-link-prefix",
        type=str,
        default="..",
        help="Relative link prefix used when --repo-url is not set.",
    )
    parser.add_argument(
        "--strings-path",
        type=Path,
        default=Path(__file__).resolve().parent / "leaderboard_strings.json",
        help="JSON file containing UI strings and metric descriptions.",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=Path(__file__).resolve().parent / "leaderboard_template.html",
        help="HTML template file for the leaderboard page.",
    )
    parsed = parser.parse_args(args)

    strings = load_strings(parsed.strings_path)
    template = load_template(parsed.template_path)
    raw_experiments = collect_experiments(parsed.results_dir)
    experiments = collapse_repeated_experiments(raw_experiments)

    repo_url = parsed.repo_url or infer_default_repo_url(strings)
    build_experiment_links(raw_experiments, repo_url, parsed.local_link_prefix)
    build_experiment_links(experiments, repo_url, parsed.local_link_prefix)

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "results_dir": str(parsed.results_dir),
        "experiments": experiments,
        "raw_experiments": raw_experiments,
        "strings": strings,
    }

    output_path = parsed.output_dir / "index.html"
    build_html(payload, output_path, template)

    icon_src = Path("docs/urb.png")
    if icon_src.exists():
        shutil.copy(icon_src, parsed.output_dir / "urb.png")

    print(f"Wrote leaderboard to {output_path}")


if __name__ == "__main__":
    main()
