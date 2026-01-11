#!/usr/bin/env python3
"""
Compute a binned prior from Horndeski sample files for a selected column.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_SAMPLES_DIR = "/Volumes/My Passport/Horndeski_samples_py_onlybackground_4"
# DEFAULT_SAMPLES_DIR = "/Users/dcz/Documents/eft_code/EFTCosmoMC/EFTCAMB/Horndeski_samples_py_onlybackground_4"
DEFAULT_OUT_DIR = (
    "/Users/dcz/Documents/eft_code/EFTCosmoMC/EFTCAMB/Horndeski_samples_piror_122602_2"
)

def normalize_token(token: str) -> str:
    token = token.strip().lower()
    return "".join(ch for ch in token if ch.isalnum())


SYNONYMS_RAW = {
    "x": ["x", "ln(a)", "lna", "ln_a", "log(a)", "loga", "log_a"],
    "a": ["a", "scale_factor", "scalefactor"],
    "z": ["z", "redshift"],
    "h2": ["h2", "mathcal{h}^2", "mathcalh2", "calh2", "h^2"],
    "wde": ["wde", "w_de", "wde(a)"],
    "omegade": ["omegade", "omega_de", "omegade", "omega de"],
}

SYNONYMS = {
    key: [normalize_token(s) for s in values] for key, values in SYNONYMS_RAW.items()
}

BASE_KEYS = ["x", "a", "z", "h2"]
DEFAULT_TARGET = "wde"


def resolve_target_key(target_name: str) -> str:
    target_norm = normalize_token(target_name)
    for key, syns in SYNONYMS.items():
        if target_norm == key or target_norm in syns:
            return key
    return target_norm


def read_header_tokens(path: str) -> List[str] | None:
    header_tokens = None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    content = stripped.lstrip("#").strip()
                    if not content:
                        continue
                    tokens = content.split()
                    if any(any(ch.isalpha() for ch in tok) for tok in tokens):
                        header_tokens = tokens
                else:
                    break
    except OSError:
        return None
    return header_tokens


def build_token_map(header_tokens: List[str]) -> Dict[str, int]:
    token_map: Dict[str, int] = {}
    for idx, tok in enumerate(header_tokens):
        norm = normalize_token(tok)
        if norm and norm not in token_map:
            token_map[norm] = idx
    return token_map


def find_column_index(token_map: Dict[str, int], key: str) -> int | None:
    for syn in SYNONYMS.get(key, [normalize_token(key)]):
        if syn in token_map:
            return token_map[syn]
    return None


def find_target_index(
    token_map: Dict[str, int], target_name: str, target_key: str
) -> int | None:
    candidates: List[str] = []
    if target_key in SYNONYMS:
        candidates.extend(SYNONYMS[target_key])
    target_norm = normalize_token(target_name)
    if target_norm:
        candidates.append(target_norm)
    if target_key:
        candidates.append(target_key)

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand in token_map:
            return token_map[cand]
    return None


def read_sample_data(
    path: str, base_keys: List[str], target_name: str, target_key: str
) -> Tuple[Dict[str, np.ndarray] | None, str | None]:
    header_tokens = read_header_tokens(path)
    if header_tokens is None:
        return None, "missing header or unreadable file"

    token_map = build_token_map(header_tokens)
    col_map: Dict[str, int | None] = {}
    for key in base_keys:
        col_map[key] = find_column_index(token_map, key)
    target_idx = find_target_index(token_map, target_name, target_key)

    missing = [key for key, idx in col_map.items() if idx is None]
    if target_idx is None:
        missing.append(f"target({target_name})")
    if missing:
        return None, f"missing columns: {', '.join(missing)}"

    try:
        data = np.loadtxt(path, comments="#")
    except Exception as exc:
        return None, f"failed to read data: {exc}"

    if data.size == 0:
        return None, "no data rows"
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 2:
        return None, "too few rows for interpolation"
    max_idx = max(
        [idx for idx in col_map.values() if idx is not None] + [target_idx]
    )
    if data.shape[1] <= max_idx:
        return None, "data row has fewer columns than header"

    sample = {}
    for key, idx in col_map.items():
        sample[key] = data[:, idx]
    sample["target"] = data[:, target_idx]
    return sample, None


def check_nonfinite(sample: Dict[str, np.ndarray], keys: List[str]) -> bool:
    for key in keys:
        if not np.all(np.isfinite(sample[key])):
            return True
    return False


def interpolate_sample(
    a: np.ndarray, values: np.ndarray, a_ref: np.ndarray
) -> Tuple[np.ndarray, bool, int, bool]:
    order = np.argsort(a)
    a_sorted = a[order]
    w_sorted = values[order]

    had_duplicates = False
    if len(a_sorted) > 1 and np.any(np.diff(a_sorted) == 0):
        had_duplicates = True
        a_unique, inv = np.unique(a_sorted, return_inverse=True)
        w_sum = np.bincount(inv, weights=w_sorted)
        w_count = np.bincount(inv)
        w_sorted = w_sum / w_count
        a_sorted = a_unique

    if len(a_sorted) == 0:
        return np.full_like(a_ref, np.nan, dtype=float), False, 0, had_duplicates

    a_min = a_sorted[0]
    a_max = a_sorted[-1]
    in_range = (a_ref >= a_min) & (a_ref <= a_max)

    if len(a_sorted) >= 2:
        try:
            w_interp = np.interp(a_ref, a_sorted, w_sorted, left=np.nan, right=np.nan)
            w_interp[~in_range] = np.nan
            return w_interp, False, 0, had_duplicates
        except Exception:
            pass

    w_interp = np.full_like(a_ref, np.nan, dtype=float)
    fallback_bins = 0
    if np.any(in_range):
        idx = np.searchsorted(a_sorted, a_ref[in_range], side="left")
        idx = np.clip(idx, 0, len(a_sorted) - 1)
        left = np.clip(idx - 1, 0, len(a_sorted) - 1)
        right = idx
        a_left = a_sorted[left]
        a_right = a_sorted[right]
        choose_right = np.abs(a_ref[in_range] - a_right) < np.abs(a_ref[in_range] - a_left)
        nearest_idx = np.where(choose_right, right, left)
        w_interp[in_range] = w_sorted[nearest_idx]
        fallback_bins = int(np.count_nonzero(in_range))
    return w_interp, True, fallback_bins, had_duplicates


def percentile_summary(values: np.ndarray, percentiles: List[float]) -> Dict[float, float]:
    return {p: float(np.percentile(values, p)) for p in percentiles}


def compute_spike_scores(target_matrix: np.ndarray, a_ref: np.ndarray) -> np.ndarray:
    ln_a = np.log(a_ref)
    dln_a = np.diff(ln_a)
    n_samples, n_bin = target_matrix.shape
    if dln_a.size == 0:
        return np.zeros(n_samples, dtype=float)

    D = (target_matrix[:, 1:] - target_matrix[:, :-1]) / dln_a[None, :]
    abs_D = np.abs(D)
    has_d = np.any(np.isfinite(abs_D), axis=1)
    d_max = np.zeros(n_samples, dtype=float)
    if np.any(has_d):
        d_max[has_d] = np.nanmax(abs_D[has_d], axis=1)

    if n_bin >= 3:
        E = (D[:, 1:] - D[:, :-1]) / dln_a[None, 1:]
        abs_E = np.abs(E)
        has_e = np.any(np.isfinite(abs_E), axis=1)
        e_max = np.zeros(n_samples, dtype=float)
        if np.any(has_e):
            e_max[has_e] = np.nanmax(abs_E[has_e], axis=1)
    else:
        e_max = np.zeros(n_samples, dtype=float)

    return np.maximum(d_max, e_max)


def plot_spike_score_diagnostics(
    spike_scores: np.ndarray, out_dir: str, output_prefix: str, bins: int
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Spike score diagnostics: failed to import matplotlib ({exc}).")
        return

    finite_scores = spike_scores[np.isfinite(spike_scores)]
    if finite_scores.size == 0:
        print("Spike score diagnostics: no finite scores, skip plot.")
        return

    positive_scores = finite_scores[finite_scores > 0]
    if positive_scores.size == 0:
        print("Spike score diagnostics: no positive scores for log10, skip.")
        return
    if positive_scores.size < finite_scores.size:
        skipped = finite_scores.size - positive_scores.size
        print(f"Spike score diagnostics: skipped {skipped} non-positive scores.")

    s_vals = np.log10(positive_scores)
    counts, edges = np.histogram(s_vals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    sorted_s = np.sort(s_vals)
    ecdf = np.arange(1, sorted_s.size + 1, dtype=float) / float(sorted_s.size)

    fig, (ax_hist, ax_ecdf) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )
    ax_hist.plot(centers, counts, color="tab:blue", linewidth=2.0)
    ax_hist.set_ylabel("Count per bin")
    ax_hist.set_title("Spike score histogram in log10 space")
    ax_hist.grid(False, alpha=0.2, linewidth=0.8)

    ax_ecdf.plot(sorted_s, ecdf, color="tab:orange", linewidth=2.0)
    ax_ecdf.set_xlabel("log10(score)")
    ax_ecdf.set_ylabel("ECDF")
    ax_ecdf.set_ylim(0.0, 1.0)
    ax_ecdf.grid(False, alpha=0.2, linewidth=0.8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{output_prefix}_spike_score_diag.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote spike score diagnostics: {out_path}")


def add_quantile(q_list: List[float], q: float, tol: float = 1e-6) -> None:
    for existing in q_list:
        if abs(existing - q) <= tol:
            return
    q_list.append(q)


def sanitize_quantiles(values: List[float]) -> List[float]:
    cleaned: List[float] = []
    for q in values:
        if 0.0 < q < 1.0:
            add_quantile(cleaned, q)
    return sorted(cleaned)


def quantile_tag(q: float) -> str:
    return f"q{q * 100:.2f}".replace(".", "p")


def corr_submatrix_diff(corr_a: np.ndarray, corr_b: np.ndarray, n_bins: int) -> float:
    if corr_a is None or corr_b is None:
        return float("nan")
    sub_a = corr_a[:n_bins, :n_bins]
    sub_b = corr_b[:n_bins, :n_bins]
    mask = np.isfinite(sub_a) & np.isfinite(sub_b)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(sub_a[mask] - sub_b[mask])))


def compute_statistics(
    target_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bin = target_matrix.shape[1]
    finite_mask = np.isfinite(target_matrix)
    n_eff = np.sum(finite_mask, axis=0)
    mean_target = np.full(n_bin, np.nan, dtype=float)
    var_target = np.full(n_bin, np.nan, dtype=float)

    for i in range(n_bin):
        if n_eff[i] > 0:
            mean_target[i] = np.nanmean(target_matrix[:, i])
        if n_eff[i] > 1:
            var_target[i] = np.nanvar(target_matrix[:, i], ddof=1)

    cov = np.full((n_bin, n_bin), np.nan, dtype=float)
    n_eff_ij = np.zeros((n_bin, n_bin), dtype=int)
    for i in range(n_bin):
        wi = target_matrix[:, i]
        fi = np.isfinite(wi)
        for j in range(i, n_bin):
            wj = target_matrix[:, j]
            mask = fi & np.isfinite(wj)
            n = int(np.count_nonzero(mask))
            n_eff_ij[i, j] = n
            n_eff_ij[j, i] = n
            if n >= 2:
                di = wi[mask] - mean_target[i]
                dj = wj[mask] - mean_target[j]
                cov_ij = float(np.dot(di, dj) / (n - 1))
                cov[i, j] = cov_ij
                cov[j, i] = cov_ij

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))

    return mean_target, var_target, cov, corr, n_eff, n_eff_ij


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute binned prior statistics for a selected column."
    )
    parser.add_argument(
        "--samples-dir",
        action="append",
        help=(
            "Directory containing sample .dat files. "
            "Can be given multiple times. "
            f"Default: {DEFAULT_SAMPLES_DIR}"
        ),
    )
    parser.add_argument(
        "--samples-dirs",
        nargs="+",
        help=(
            "Space-separated list of sample directories. "
            "Merged with --samples-dir if both are provided."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--pattern",
        default="*.dat",
        help="Glob pattern for sample files.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Target column name to analyze (default: wDE).",
    )
    parser.add_argument(
        "--n-bin",
        type=int,
        default=100,
        help="Number of bins in a.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=6.0,
        help="Maximum redshift for binning (a_min = 1/(1+z_max)).",
    )
    parser.add_argument(
        "--low-eff-frac",
        type=float,
        default=0.3,
        help="Threshold fraction for low N_eff warnings.",
    )
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=0.0,
        help=(
            "Deprecated: if >0, use apply-quantile=1-trim-frac "
            "and include it in diagnostics."
        ),
    )
    parser.add_argument(
        "--winsorize",
        action="store_true",
        help=(
            "Deprecated: ignored for spike-based trimming "
            "(kept for backward compatibility)."
        ),
    )
    parser.add_argument(
        "--score-bins",
        type=int,
        default=80,
        help="Number of bins for spike score histogram in log10(score) space.",
    )
    parser.add_argument(
        "--score-quantiles",
        nargs="+",
        type=float,
        default=[0.99, 0.995, 0.999],
        help=(
            "Quantiles for spike score thresholds (e.g. 0.99 0.995 0.999). "
            "Each quantile writes a correlation matrix for diagnostics."
        ),
    )
    parser.add_argument(
        "--apply-quantile",
        type=float,
        default=None,
        help=(
            "Quantile threshold to apply for final trimming. "
            "Defaults to max(score-quantiles)."
        ),
    )
    parser.add_argument(
        "--low-a-bins",
        type=int,
        default=0,
        help=(
            "Number of lowest-a bins for correlation stability diagnostics "
            "(0 uses max(5, n_bin//10))."
        ),
    )
    args = parser.parse_args()

    samples_dirs: List[str] = []
    if args.samples_dir:
        samples_dirs.extend(args.samples_dir)
    if args.samples_dirs:
        samples_dirs.extend(args.samples_dirs)
    if not samples_dirs:
        samples_dirs = [DEFAULT_SAMPLES_DIR]
    samples_dirs = list(dict.fromkeys(samples_dirs))

    out_dir = args.out_dir or samples_dirs[0]

    target_name = args.target
    target_key = resolve_target_key(target_name)
    if not target_key:
        print("ERROR: target name resolves to an empty token.")
        return 1
    output_prefix = target_key
    key_checks = BASE_KEYS + ["target"]

    if args.trim_frac < 0 or args.trim_frac >= 1.0:
        print("ERROR: trim-frac must be in [0, 1).")
        return 1
    if args.apply_quantile is not None and not (0.0 < args.apply_quantile < 1.0):
        print("ERROR: apply-quantile must be in (0, 1).")
        return 1

    print(f"Target column: {target_name} (prefix={output_prefix})")
    print("Step 0: reading sample files")
    missing_dirs = [path for path in samples_dirs if not os.path.isdir(path)]
    if missing_dirs:
        print("ERROR: samples directory not found:")
        for path in missing_dirs:
            print(f"  {path}")
        return 1

    files: List[str] = []
    patterns: List[str] = []
    for samples_dir in samples_dirs:
        pattern = os.path.join(samples_dir, args.pattern)
        patterns.append(pattern)
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        print("ERROR: no files found with patterns:")
        for pattern in patterns:
            print(f"  {pattern}")
        return 1

    print(f"Found {len(files)} files from {len(samples_dirs)} directories.")
    read_samples: List[Dict[str, np.ndarray]] = []
    read_names: List[str] = []
    read_paths: List[str] = []
    skip_reasons_step0: List[Tuple[str, str]] = []

    for path in files:
        sample, reason = read_sample_data(path, BASE_KEYS, target_name, target_key)
        if reason is not None:
            skip_reasons_step0.append((path, reason))
            print(f"SKIP step0: {path} -> {reason}")
            continue
        read_samples.append(sample)
        read_names.append(os.path.basename(path))
        read_paths.append(path)

    print(
        f"Step 0 summary: total={len(files)} read_ok={len(read_samples)} "
        f"skipped={len(skip_reasons_step0)}"
    )

    print("Step 1: data quality checks for key columns")
    good_samples: List[Dict[str, np.ndarray]] = []
    good_names: List[str] = []
    good_paths: List[str] = []
    skip_reasons_step1: List[Tuple[str, str]] = []
    for sample, name, path in zip(read_samples, read_names, read_paths):
        if check_nonfinite(sample, key_checks):
            skip_reasons_step1.append((path, "non-finite values in key columns"))
            print(f"SKIP step1: {path} -> non-finite values in key columns")
            continue
        good_samples.append(sample)
        good_names.append(name)
        good_paths.append(path)

    total_good = len(good_samples)
    print(
        f"Step 1 summary: total={len(files)} read_ok={len(read_samples)} "
        f"passed_quality={total_good} skipped_quality={len(skip_reasons_step1)}"
    )
    if total_good == 0:
        print("ERROR: no samples passed quality checks.")
        return 1

    os.makedirs(out_dir, exist_ok=True)
    skip_report_path = os.path.join(out_dir, f"{output_prefix}_skipped_samples.txt")
    if skip_reasons_step0 or skip_reasons_step1:
        with open(skip_report_path, "w", encoding="utf-8") as handle:
            handle.write("step\tpath\treason\n")
            for path, reason in skip_reasons_step0:
                handle.write(f"0\t{path}\t{reason}\n")
            for path, reason in skip_reasons_step1:
                handle.write(f"1\t{path}\t{reason}\n")
        print(f"Wrote skip report: {skip_report_path}")

    print("Step 2: building reference bins")
    if args.n_bin <= 0:
        print("ERROR: n-bin must be positive.")
        return 1
    if args.z_max <= 0:
        print("ERROR: z-max must be positive.")
        return 1

    a_min = 1.0 / (1.0 + args.z_max)
    a_max = 1.0
    delta_a = (a_max - a_min) / args.n_bin
    idx = np.arange(args.n_bin, dtype=float)
    a_ref = a_min + (idx + 0.5) * delta_a
    z_ref = 1.0 / a_ref - 1.0
    print(
        f"Reference bins: n_bin={args.n_bin} a_min={a_min:.8f} "
        f"a_max={a_max:.8f} delta_a={delta_a:.8f}"
    )
    print(
        f"Reference points: a[0]={a_ref[0]:.8f} a[-1]={a_ref[-1]:.8f} "
        f"z[0]={z_ref[0]:.8f} z[-1]={z_ref[-1]:.8f}"
    )

    print(f"Step 3: interpolating {target_name} onto reference bins")
    target_matrix = np.full((total_good, args.n_bin), np.nan, dtype=float)
    fallback_samples = 0
    fallback_bins_total = 0
    duplicate_samples = 0
    sample_a_ranges = np.zeros((total_good, 2), dtype=float)
    coverage_count = np.zeros(args.n_bin, dtype=int)

    for s_idx, sample in enumerate(good_samples):
        a_vals = sample["a"]
        target_vals = sample["target"]
        target_interp, used_fallback, fallback_bins, had_duplicates = interpolate_sample(
            a_vals, target_vals, a_ref
        )
        target_matrix[s_idx, :] = target_interp

        a_min_s = float(np.min(a_vals))
        a_max_s = float(np.max(a_vals))
        sample_a_ranges[s_idx, :] = [a_min_s, a_max_s]
        coverage_mask = (a_ref >= a_min_s) & (a_ref <= a_max_s)
        coverage_count += coverage_mask.astype(int)

        if used_fallback:
            fallback_samples += 1
            fallback_bins_total += fallback_bins
        if had_duplicates:
            duplicate_samples += 1

    print(
        "Interpolation summary: "
        f"fallback_samples={fallback_samples} "
        f"fallback_bins={fallback_bins_total} "
        f"duplicate_a_samples={duplicate_samples}"
    )
    coverage_stats = {
        "min": int(np.min(coverage_count)),
        "median": int(np.median(coverage_count)),
        "max": int(np.max(coverage_count)),
    }
    print(f"Coverage count stats (bins in range per bin): {coverage_stats}")

    print("Step 3b: spike score diagnostics and quantile-based trimming")
    if args.winsorize:
        print("Note: --winsorize ignored for spike-based trimming.")

    spike_scores = compute_spike_scores(target_matrix, a_ref)
    if args.score_bins > 0:
        plot_spike_score_diagnostics(
            spike_scores, out_dir, output_prefix, args.score_bins
        )
    else:
        print("Spike score diagnostics: --score-bins <= 0, skip plot.")

    score_sort_path = os.path.join(out_dir, f"{output_prefix}_spike_scores_sorted.txt")
    score_sort_key = np.where(np.isfinite(spike_scores), spike_scores, np.inf)
    sorted_idx = np.argsort(score_sort_key)
    with open(score_sort_path, "w", encoding="utf-8") as handle:
        handle.write("rank\tindex\tname\tspike_score\n")
        for rank, idx in enumerate(sorted_idx, start=1):
            handle.write(
                f"{rank}\t{idx}\t{good_names[idx]}\t{spike_scores[idx]:.8e}\n"
            )
    print(f"Wrote spike score ranking: {score_sort_path}")

    finite_scores = spike_scores[np.isfinite(spike_scores)]
    apply_quantile = args.apply_quantile
    quantile_levels = sanitize_quantiles(args.score_quantiles or [])
    if args.trim_frac > 0:
        q_from_trim = 1.0 - args.trim_frac
        if not (0.0 < q_from_trim < 1.0):
            print("ERROR: trim-frac implies invalid quantile.")
            return 1
        print(
            "WARNING: --trim-frac is deprecated; "
            f"using apply-quantile={q_from_trim:.6f}."
        )
        add_quantile(quantile_levels, q_from_trim)
        if apply_quantile is None:
            apply_quantile = q_from_trim

    if apply_quantile is None and quantile_levels:
        apply_quantile = max(quantile_levels)
    if apply_quantile is not None:
        add_quantile(quantile_levels, apply_quantile)
    quantile_levels = sorted(quantile_levels)

    thresholds: Dict[float, float] = {}
    if finite_scores.size == 0:
        print("WARNING: no finite spike scores; skipping trimming diagnostics.")
        apply_quantile = None
    elif quantile_levels:
        for q in quantile_levels:
            thresholds[q] = float(np.quantile(finite_scores, q))

        low_a_bins = args.low_a_bins if args.low_a_bins > 0 else max(5, args.n_bin // 10)
        low_a_bins = min(low_a_bins, args.n_bin)

        _, _, _, base_corr, _, _ = compute_statistics(target_matrix)
        diag_path = os.path.join(out_dir, f"{output_prefix}_spike_trim_diagnostics.txt")
        with open(diag_path, "w", encoding="utf-8") as handle:
            handle.write(
                "quantile\tthreshold\ttrimmed\tkept\tdiff_vs_base\tdiff_vs_prev\n"
            )
            prev_corr = None
            for q in quantile_levels:
                threshold = thresholds[q]
                keep_mask_q = np.isfinite(spike_scores) & (spike_scores <= threshold)
                kept = int(np.count_nonzero(keep_mask_q))
                trimmed = total_good - kept
                if kept < 2:
                    corr_q = np.full((args.n_bin, args.n_bin), np.nan, dtype=float)
                else:
                    _, _, _, corr_q, _, _ = compute_statistics(
                        target_matrix[keep_mask_q, :]
                    )
                corr_path = os.path.join(
                    out_dir, f"{output_prefix}_correlation_{quantile_tag(q)}.txt"
                )
                np.savetxt(corr_path, corr_q, header="R_ij")
                diff_base = corr_submatrix_diff(corr_q, base_corr, low_a_bins)
                diff_prev = corr_submatrix_diff(corr_q, prev_corr, low_a_bins)
                handle.write(
                    f"{q:.6f}\t{threshold:.8e}\t{trimmed}\t{kept}\t"
                    f"{diff_base:.6e}\t{diff_prev:.6e}\n"
                )
                prev_corr = corr_q

        print(f"Wrote spike trim diagnostics: {diag_path}")
        print(f"Low-a bins for stability check: {low_a_bins}")
        print("Spike score quantile thresholds:")
        for q in quantile_levels:
            threshold = thresholds[q]
            keep_mask_q = np.isfinite(spike_scores) & (spike_scores <= threshold)
            kept = int(np.count_nonzero(keep_mask_q))
            trimmed = total_good - kept
            print(
                f"  q={q:.6f} threshold={threshold:.6e} trimmed={trimmed} kept={kept}"
            )

    if apply_quantile is not None and finite_scores.size > 0:
        apply_threshold = thresholds.get(
            apply_quantile, float(np.quantile(finite_scores, apply_quantile))
        )
        keep_mask = np.isfinite(spike_scores) & (spike_scores <= apply_threshold)
        print(
            "Spike trimming applied: "
            f"apply_quantile={apply_quantile:.6f} threshold={apply_threshold:.6e}"
        )
    else:
        keep_mask = np.ones(total_good, dtype=bool)
        apply_quantile = None
        print("Spike trimming applied: none")

    trimmed_total = int(np.count_nonzero(~keep_mask))
    kept_total = int(np.count_nonzero(keep_mask))
    print(f"Spike trimming summary: trimmed={trimmed_total} kept={kept_total}")

    spike_mask_path = os.path.join(out_dir, f"{output_prefix}_spike_mask.txt")
    with open(spike_mask_path, "w", encoding="utf-8") as handle:
        handle.write("index\tname\tspike_score\tkeep\n")
        for idx, (name, score, keep) in enumerate(
            zip(good_names, spike_scores, keep_mask)
        ):
            handle.write(f"{idx}\t{name}\t{score:.8e}\t{int(keep)}\n")
    print(f"Wrote spike mask: {spike_mask_path}")

    if trimmed_total > 0:
        trim_idx = np.where(~keep_mask)[0]
        trim_sorted = trim_idx[np.argsort(spike_scores[trim_idx])[::-1]]
        print("Trimmed curves (highest spike scores):")
        for rank, idx in enumerate(trim_sorted[:10], start=1):
            print(
                f"  rank={rank:02d} idx={idx:04d} name={good_names[idx]} "
                f"score={spike_scores[idx]:.6e}"
            )
        if trimmed_total > 10:
            print(f"  ... ({trimmed_total - 10} more)")

        target_matrix = target_matrix[keep_mask, :]
        good_names = [name for name, keep in zip(good_names, keep_mask) if keep]
        good_paths = [path for path, keep in zip(good_paths, keep_mask) if keep]
        sample_a_ranges = sample_a_ranges[keep_mask, :]
        total_good = target_matrix.shape[0]
        coverage_count = np.zeros(args.n_bin, dtype=int)
        for a_min_s, a_max_s in sample_a_ranges:
            coverage_mask = (a_ref >= a_min_s) & (a_ref <= a_max_s)
            coverage_count += coverage_mask.astype(int)

    print(f"Step 4: computing mean, variance, and covariance for {target_name}")
    mean_target, var_target, cov, corr, n_eff, n_eff_ij = compute_statistics(
        target_matrix
    )

    print("N_eff per bin:")
    for i in range(args.n_bin):
        print(
            f"  bin={i:03d} a={a_ref[i]:.8f} z={z_ref[i]:.8f} "
            f"N_eff={n_eff[i]}"
        )

    n_eff_values = n_eff_ij.flatten()
    percentiles = [0, 5, 50, 95, 100]
    n_eff_summary = percentile_summary(n_eff_values, percentiles)
    print("N_eff_ij percentiles:")
    for p in percentiles:
        print(f"  p{int(p):02d}={n_eff_summary[p]}")

    print("Step 5: writing outputs")
    bins_table_path = os.path.join(out_dir, f"{output_prefix}_prior_bins.txt")
    cov_path = os.path.join(out_dir, f"{output_prefix}_covariance.txt")
    corr_path = os.path.join(out_dir, f"{output_prefix}_correlation.txt")
    n_eff_ij_path = os.path.join(out_dir, f"{output_prefix}_neff_ij.txt")

    table = np.column_stack([a_ref, z_ref, mean_target, n_eff])
    np.savetxt(
        bins_table_path,
        table,
        header=f"a_i z_i mean_{output_prefix} N_eff",
    )
    np.savetxt(cov_path, cov, header="C_ij")
    np.savetxt(corr_path, corr, header="R_ij")
    np.savetxt(n_eff_ij_path, n_eff_ij, header="N_eff_ij")

    print(f"Wrote bins table: {bins_table_path}")
    print(f"Wrote covariance matrix: {cov_path}")
    print(f"Wrote correlation matrix: {corr_path}")
    print(f"Wrote N_eff_ij matrix: {n_eff_ij_path}")

    low_eff_threshold = int(np.ceil(args.low_eff_frac * total_good))
    low_eff_bins = np.where(n_eff < low_eff_threshold)[0]
    if low_eff_bins.size > 0:
        print(
            "WARNING: low N_eff bins detected "
            f"(threshold={low_eff_threshold} of total={total_good})"
        )
        for i in low_eff_bins:
            coverage_frac = coverage_count[i] / float(total_good)
            print(
                f"  bin={i:03d} a={a_ref[i]:.8f} z={z_ref[i]:.8f} "
                f"N_eff={n_eff[i]} coverage_frac={coverage_frac:.3f} "
                "reason_hint=coverage_or_interpolation"
            )
        print(
            "Possible reasons: many samples do not cover this a range, "
            "interpolation fallback or failures, or quality-filtered samples."
        )

    print("Step 6: stability checks for covariance and correlation")
    diag = np.diag(cov)
    diag_problem_bins = [
        i for i in range(args.n_bin) if (not np.isfinite(diag[i])) or diag[i] <= 0
    ]

    corr_nan_pairs: List[Tuple[int, int]] = []
    for i in range(args.n_bin):
        for j in range(i, args.n_bin):
            if n_eff_ij[i, j] >= 2 and not np.isfinite(corr[i, j]):
                corr_nan_pairs.append((i, j))

    if diag_problem_bins or corr_nan_pairs:
        report_path = os.path.join(out_dir, f"{output_prefix}_instability_report.txt")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("Instability report\n")
            handle.write("Problematic diagonal bins:\n")
            if diag_problem_bins:
                for i in diag_problem_bins:
                    handle.write(
                        f"  bin={i:03d} a={a_ref[i]:.8f} z={z_ref[i]:.8f} "
                        f"C_ii={diag[i]} N_eff={n_eff[i]}\n"
                    )
                    contributors = [
                        good_names[s]
                        for s in range(total_good)
                        if np.isfinite(target_matrix[s, i])
                    ]
                    handle.write(f"    contributors={contributors}\n")
            else:
                handle.write("  none\n")

            handle.write("Problematic correlation pairs:\n")
            if corr_nan_pairs:
                for i, j in corr_nan_pairs:
                    handle.write(
                        f"  pair=({i:03d},{j:03d}) "
                        f"a_i={a_ref[i]:.8f} a_j={a_ref[j]:.8f} "
                        f"N_eff_ij={n_eff_ij[i, j]}\n"
                    )
                    contributors = [
                        good_names[s]
                        for s in range(total_good)
                        if np.isfinite(target_matrix[s, i])
                        and np.isfinite(target_matrix[s, j])
                    ]
                    handle.write(f"    contributors={contributors}\n")
            else:
                handle.write("  none\n")

            handle.write("Suggestions:\n")
            handle.write(
                "  - Check sample coverage in a for problematic bins.\n"
                "  - Ensure interpolation has >=2 points per sample in range.\n"
                "  - Verify target values are not constant across samples.\n"
                "  - Review any filtering (e.g. OmegaDE thresholds) that could "
                "remove data.\n"
            )

        print(
            "WARNING: instability detected (C_ii<=0 or NaN correlation). "
            f"Details written to {report_path}"
        )
        print(
            f"Problematic bins: {diag_problem_bins if diag_problem_bins else 'none'}"
        )
        if corr_nan_pairs:
            print(f"Problematic correlation pairs (count={len(corr_nan_pairs)}).")
        else:
            print("Problematic correlation pairs: none.")
    else:
        print("No instability detected in covariance or correlation matrices.")

    print(f"Effective samples used: {total_good}")
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
