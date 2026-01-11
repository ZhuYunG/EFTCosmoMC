#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _detect_header_and_delim(header_line: str):
    header_line = header_line.strip("\n")
    if "\t" in header_line:
        return header_line.split("\t"), "\t"
    return header_line.split(), None


def _split_row(line: str, header_len: int, delim: Optional[str]):
    if delim is None:
        return line.strip().split(maxsplit=header_len - 1)
    parts = line.rstrip("\n").split(delim)
    if len(parts) > header_len:
        tail = delim.join(parts[header_len - 1 :])
        parts = parts[: header_len - 1] + [tail]
    return parts


def _strip_outer_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _is_success(value: str) -> bool:
    raw = str(value).strip().lower()
    if raw in {"true", "t", "yes", "y"}:
        return True
    try:
        return float(raw) > 0.5
    except ValueError:
        return False


def _to_float(value, name: str, line_num: int, context: str):
    try:
        num = float(value)
    except (TypeError, ValueError):
        _warn(f"{context}line {line_num}: {name} is not numeric: {value!r}")
        return None
    if math.isnan(num) or math.isinf(num):
        _warn(f"{context}line {line_num}: {name} is NaN/inf")
        return None
    return num


def _get_param_value(row, params_json, name: str, line_num: int, context: str):
    if name in row and row[name] != "":
        return _to_float(row[name], name, line_num, context)
    if name in params_json:
        return _to_float(params_json[name], name, line_num, context)
    _warn(f"{context}line {line_num}: missing parameter {name!r}")
    return None


def read_params(input_path: str, x_param: str, y_param: str):
    x_vals = []
    y_vals = []
    accepted = 0
    total = 0
    context = f"{input_path}: "

    with open(input_path, "r", encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"{input_path}: input file is empty")
        header, delim = _detect_header_and_delim(header_line)
        header_len = len(header)
        if "params_json" not in header:
            _warn(
                f"{input_path}: header missing params_json column; only top-level columns will be used"
            )

        for line_num, line in enumerate(handle, start=2):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            total += 1
            parts = _split_row(line, header_len, delim)
            if len(parts) != header_len:
                _warn(
                    f"{context}line {line_num}: expected {header_len} columns, got {len(parts)}"
                )
                continue
            row = dict(zip(header, parts))
            if not _is_success(row.get("success", "")):
                continue
            accepted += 1

            params_json = {}
            json_raw = row.get("params_json", "")
            if json_raw:
                json_raw = _strip_outer_quotes(json_raw)
                try:
                    params_json = json.loads(json_raw)
                except json.JSONDecodeError as exc:
                    _warn(f"{context}line {line_num}: params_json decode error: {exc}")
                    continue

            x_val = _get_param_value(row, params_json, x_param, line_num, context)
            y_val = _get_param_value(row, params_json, y_param, line_num, context)
            if x_val is None or y_val is None:
                continue
            x_vals.append(x_val)
            y_vals.append(y_val)

    return x_vals, y_vals, accepted, total


def read_params_multi(input_paths, x_param: str, y_param: str):
    all_x = []
    all_y = []
    accepted_total = 0
    total_rows = 0

    for input_path in input_paths:
        x_vals, y_vals, accepted, total = read_params(input_path, x_param, y_param)
        all_x.extend(x_vals)
        all_y.extend(y_vals)
        accepted_total += accepted
        total_rows += total

    return all_x, all_y, accepted_total, total_rows


def plot_scatter(
    x_vals, y_vals, x_param: str, y_param: str, out_path: str, title: Optional[str]
):
    if not x_vals:
        raise ValueError("no valid accepted samples to plot")

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(x_vals, y_vals, s=10, alpha=0.6, edgecolors="none")
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(title or f"{x_param} vs {y_param} (accepted samples)")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot scatter of two parameters from accepted samples.")
    parser.add_argument(
        "--input", required=True, nargs="+", help="Path(s) to input txt file(s)."
    )
    parser.add_argument("--xparam", required=True, help="Parameter name for x-axis.")
    parser.add_argument("--yparam", required=True, help="Parameter name for y-axis.")
    parser.add_argument("--outdir", required=True, help="Output directory for the PDF plot.")
    parser.add_argument("--outname", default=None, help="Output PDF filename.")
    parser.add_argument("--title", default=None, help="Custom plot title.")
    args = parser.parse_args()

    out_name = args.outname or f"{args.xparam}_{args.yparam}.pdf"
    out_path = os.path.join(args.outdir, out_name)
    os.makedirs(args.outdir, exist_ok=True)

    x_vals, y_vals, accepted, total = read_params_multi(
        args.input, args.xparam, args.yparam
    )
    plot_scatter(x_vals, y_vals, args.xparam, args.yparam, out_path, args.title)

    print(
        "saved "
        f"{out_path} (inputs: {len(args.input)}, accepted samples: {accepted}, total rows: {total}, points: {len(x_vals)})"
    )


if __name__ == "__main__":
    main()
