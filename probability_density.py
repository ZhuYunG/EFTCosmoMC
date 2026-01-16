#!/usr/bin/env python3
"""
Build an ensemble probability density plot for wDE from .dat files.
"""
from __future__ import annotations

from pathlib import Path
import csv
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# -------------------------
# Editable parameters
# -------------------------
folder = "EFTCAMB"
folders = ["/Users/dcz/data/Horndeski_samples/Horndeski_samples_py_onlybackground_a01_1"]  # If non-empty, read from multiple folders (overrides `folder`)
pattern = "Horndeski_sample_*.dat"
out_folder = "/Users/dcz/data/plots_a01_1"

use_a = False  # False -> use z (col 2), True -> use a (col 1)
zmin = 0.1  # e.g. 0.0; use None to disable
zmax = 6.0  # e.g. 2.0; use None to disable

wmin, wmax = -2.0, 1.0
n_wbin = 250

smooth_sigma = 0.15  # set >0 to enable Gaussian smoothing in w direction
use_log = False  # optional log color scale
reverse_cmap = False
add_colorbar = False
show_legend = False
x_log10 = True  # log10 scale for x-axis (requires positive x)

dpi = 500
write_quantiles = True
log_vmin = None  # set when use_log=True

figsize = (7.5, 4.8)

# Trim options (spike-based: max |w| per sample)
trim_frac = 0.15  # drop top fraction by spike score; 0 disables
trim_quantile = None  # e.g. 0.995; ignored if trim_frac > 0
write_trim_report = True

# Weight options
weight_csv = "/Users/dcz/data/weights/Horndeski_samples_py_onlybackground_a01_1.csv"  # path or list of paths; CSV has N columns, see weight_col
weight_col = 12  # 0-based index; 8 -> 9th column
weight_key_mode = "auto"  # "auto", "path", or "basename"
missing_weight = "skip"  # "skip", "unity", or "error"
# -------------------------


def compute_edges(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be 1D with length >= 2")
    mid = 0.5 * (grid[1:] + grid[:-1])
    edges = np.empty(grid.size + 1, dtype=grid.dtype)
    edges[1:-1] = mid
    edges[0] = grid[0] - (mid[0] - grid[0])
    edges[-1] = grid[-1] + (grid[-1] - mid[-1])
    return edges


def normalize_folders(folder: str, folders: list[str] | str | Path | None) -> list[Path]:
    if folders is None:
        return [Path(folder)]
    if isinstance(folders, (str, Path)):
        return [Path(folders)]
    if len(folders) == 0:
        return [Path(folder)]
    return [Path(f) for f in folders]


def collect_paths(folder_paths: list[Path], pattern: str) -> list[Path]:
    paths: list[Path] = []
    for folder_path in folder_paths:
        paths.extend(folder_path.glob(pattern))
    return sorted(set(paths))


def normalize_csv_paths(paths: list[str] | str | Path | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def load_weight_map(
    csv_paths: list[Path], weight_col: int
) -> tuple[dict[str, float], dict[str, float | None], dict]:
    path_map: dict[str, float] = {}
    base_map: dict[str, float | None] = {}
    base_src: dict[str, str] = {}
    stats = {
        "rows": 0,
        "bad_cols": 0,
        "bad_weight": 0,
        "empty_key": 0,
        "duplicate_key": 0,
        "ambiguous_base": 0,
        "missing_file": 0,
    }

    for csv_path in csv_paths:
        if not csv_path.exists():
            stats["missing_file"] += 1
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                if row[0].lstrip().startswith("#"):
                    continue
                stats["rows"] += 1
                if len(row) <= weight_col:
                    stats["bad_cols"] += 1
                    continue
                key = row[0].strip()
                if not key:
                    stats["empty_key"] += 1
                    continue
                try:
                    weight = float(row[weight_col])
                except ValueError:
                    stats["bad_weight"] += 1
                    continue
                if key in path_map:
                    stats["duplicate_key"] += 1
                path_map[key] = weight
                base = Path(key).name
                if base in base_src and base_src[base] != key:
                    base_map[base] = None
                elif base not in base_map:
                    base_map[base] = weight
                    base_src[base] = key

    stats["ambiguous_base"] = sum(1 for v in base_map.values() if v is None)
    return path_map, base_map, stats


def match_weights(
    paths: list[Path],
    path_map: dict[str, float],
    base_map: dict[str, float | None],
    key_mode: str,
    missing_policy: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    weights = np.full(len(paths), np.nan, dtype=float)
    stats = {
        "matched_path": 0,
        "matched_basename": 0,
        "missing": 0,
        "ambiguous_basename": 0,
        "bad_weight": 0,
    }

    for i, path in enumerate(paths):
        weight = None
        key = str(path)
        key_resolved = str(path.resolve())
        if key_mode in ("auto", "path"):
            if key in path_map:
                weight = path_map[key]
                stats["matched_path"] += 1
            elif key_resolved in path_map:
                weight = path_map[key_resolved]
                stats["matched_path"] += 1
        if weight is None and key_mode in ("auto", "basename"):
            base = path.name
            if base in base_map:
                if base_map[base] is None:
                    stats["ambiguous_basename"] += 1
                else:
                    weight = base_map[base]
                    stats["matched_basename"] += 1

        if weight is None:
            stats["missing"] += 1
        else:
            weights[i] = weight

    bad_mask = ~np.isfinite(weights) | (weights < 0)
    stats["bad_weight"] = int(np.count_nonzero(bad_mask))
    if stats["bad_weight"] > 0:
        weights[bad_mask] = np.nan

    if missing_policy == "unity":
        weights[np.isnan(weights)] = 1.0
        keep_mask = np.ones(len(paths), dtype=bool)
    elif missing_policy == "error":
        if np.isnan(weights).any():
            raise ValueError("Missing or invalid weights with missing_weight='error'.")
        keep_mask = np.ones(len(paths), dtype=bool)
    else:
        keep_mask = np.isfinite(weights)

    return weights, keep_mask, stats


def load_wde_samples(
    paths: list[Path], use_a: bool
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict, list[Path]]:
    grid_col = 1 if use_a else 2
    z_col = 2
    w_col = 17
    skip = {
        "load_error": 0,
        "empty": 0,
        "ndim": 0,
        "cols": 0,
        "nonfinite": 0,
        "grid_len": 0,
        "grid_mismatch": 0,
        "z_grid_len": 0,
        "z_grid_mismatch": 0,
    }

    grid_ref = None
    z_ref = None
    w_list: list[np.ndarray] = []
    good_paths: list[Path] = []

    for path in paths:
        try:
            data = np.loadtxt(path, comments="#")
        except Exception:
            skip["load_error"] += 1
            continue

        if data.size == 0:
            skip["empty"] += 1
            continue

        if data.ndim != 2:
            skip["ndim"] += 1
            continue

        if data.shape[1] <= w_col:
            skip["cols"] += 1
            continue

        grid = data[:, grid_col]
        z_grid = data[:, z_col]
        w = data[:, w_col]

        # Only require finite values in the columns we actually use.
        if not (np.isfinite(grid).all() and np.isfinite(z_grid).all() and np.isfinite(w).all()):
            skip["nonfinite"] += 1
            continue

        if grid_ref is None:
            grid_ref = grid.copy()
            z_ref = z_grid.copy()
        else:
            if grid.shape[0] != grid_ref.shape[0]:
                skip["grid_len"] += 1
                continue
            if not np.allclose(grid, grid_ref, rtol=1e-8, atol=1e-10):
                skip["grid_mismatch"] += 1
                continue
            if z_grid.shape[0] != z_ref.shape[0]:
                skip["z_grid_len"] += 1
                continue
            if not np.allclose(z_grid, z_ref, rtol=1e-8, atol=1e-10):
                skip["z_grid_mismatch"] += 1
                continue

        w_list.append(w)
        good_paths.append(path)

    if grid_ref is None or z_ref is None or not w_list:
        return None, None, None, skip, []

    W = np.vstack(w_list)
    return W, grid_ref, z_ref, skip, good_paths


def build_density(W: np.ndarray, w_bins: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_grid = W.shape[1]
    n_wbin = len(w_bins) - 1
    D = np.zeros((n_grid, n_wbin), dtype=float)
    for i in range(n_grid):
        vals = W[:, i]
        mask = np.isfinite(vals) & np.isfinite(weights) & (weights >= 0)
        if not np.any(mask):
            continue
        pdf, _ = np.histogram(
            vals[mask], bins=w_bins, weights=weights[mask], density=True
        )
        D[i, :] = pdf
    return D


def smooth_density(D: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return D
    try:
        from scipy.ndimage import gaussian_filter1d
    except Exception:
        print("scipy not available; skipping smoothing.")
        return D
    return gaussian_filter1d(D, sigma=sigma, axis=1, mode="nearest")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.full_like(qs, np.nan, dtype=float)
    sorter = np.argsort(values)
    v_sorted = values[sorter]
    w_sorted = weights[sorter]
    cdf = np.cumsum(w_sorted)
    if cdf[-1] <= 0:
        return np.full_like(qs, np.nan, dtype=float)
    cdf /= cdf[-1]
    return np.interp(qs, cdf, v_sorted)


def compute_stats(W: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, ...]:
    n_grid = W.shape[1]
    mean = np.full(n_grid, np.nan, dtype=float)
    q16 = np.full(n_grid, np.nan, dtype=float)
    q84 = np.full(n_grid, np.nan, dtype=float)
    q025 = np.full(n_grid, np.nan, dtype=float)
    q975 = np.full(n_grid, np.nan, dtype=float)
    q005 = np.full(n_grid, np.nan, dtype=float)
    q995 = np.full(n_grid, np.nan, dtype=float)

    qs = np.array([0.16, 0.84, 0.025, 0.975, 0.005, 0.995], dtype=float)
    for i in range(n_grid):
        vals = W[:, i]
        mask = np.isfinite(vals) & np.isfinite(weights) & (weights >= 0)
        if not np.any(mask):
            continue
        v = vals[mask]
        w = weights[mask]
        w_sum = np.sum(w)
        if w_sum <= 0:
            continue
        mean[i] = np.sum(v * w) / w_sum
        q_vals = weighted_quantile(v, w, qs)
        q16[i], q84[i], q025[i], q975[i], q005[i], q995[i] = q_vals

    return mean, q16, q84, q025, q975, q005, q995


def apply_z_range(
    grid: np.ndarray,
    z_grid: np.ndarray,
    W: np.ndarray,
    zmin: float | None,
    zmax: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if zmin is None and zmax is None:
        return grid, z_grid, W
    mask = np.ones(z_grid.shape[0], dtype=bool)
    if zmin is not None:
        mask &= z_grid >= zmin
    if zmax is not None:
        mask &= z_grid <= zmax
    kept = int(np.count_nonzero(mask))
    print(f"Apply z-range: zmin={zmin} zmax={zmax} kept_bins={kept}/{z_grid.size}")
    return grid[mask], z_grid[mask], W[:, mask]


def compute_spike_scores(W: np.ndarray) -> np.ndarray:
    abs_vals = np.abs(W)
    n_samples = abs_vals.shape[0]
    scores = np.full(n_samples, np.nan, dtype=float)
    has_finite = np.any(np.isfinite(abs_vals), axis=1)
    if np.any(has_finite):
        scores[has_finite] = np.nanmax(abs_vals[has_finite], axis=1)
    return scores


def apply_trimming(
    W: np.ndarray,
    weights: np.ndarray,
    paths: list[Path],
    trim_frac: float,
    trim_quantile: float | None,
) -> tuple[np.ndarray, np.ndarray, list[Path], np.ndarray, np.ndarray]:
    if trim_frac < 0 or trim_frac >= 1.0:
        raise ValueError("trim_frac must be in [0, 1).")
    if trim_quantile is not None and not (0.0 < trim_quantile < 1.0):
        raise ValueError("trim_quantile must be in (0, 1).")
    if trim_frac > 0 and trim_quantile is not None:
        print("Note: trim_quantile ignored because trim_frac > 0.")

    scores = compute_spike_scores(W)
    finite_scores = scores[np.isfinite(scores)]
    n_samples = W.shape[0]
    keep_mask = np.ones(n_samples, dtype=bool)

    if trim_frac > 0:
        if finite_scores.size == 0:
            print("Spike trimming requested but no finite scores; skipping.")
        else:
            n_trim = int(np.floor(trim_frac * n_samples))
            if n_trim > 0:
                sort_key = np.where(np.isfinite(scores), scores, np.inf)
                trim_order = np.argsort(sort_key)
                trim_idx = trim_order[-n_trim:]
                keep_mask[trim_idx] = False
            print(f"Spike trimming applied: trim_frac={trim_frac:.6f}")
    elif trim_quantile is not None:
        if finite_scores.size == 0:
            print("Spike trimming requested but no finite scores; skipping.")
        else:
            threshold = float(np.quantile(finite_scores, trim_quantile))
            keep_mask = np.isfinite(scores) & (scores <= threshold)
            print(
                "Spike trimming applied: "
                f"trim_quantile={trim_quantile:.6f} threshold={threshold:.6e}"
            )
    else:
        print("Spike trimming applied: none")

    trimmed_total = int(np.count_nonzero(~keep_mask))
    kept_total = int(np.count_nonzero(keep_mask))
    print(f"Spike trimming summary: trimmed={trimmed_total} kept={kept_total}")

    kept_paths = [p for p, keep in zip(paths, keep_mask) if keep]
    return W[keep_mask], weights[keep_mask], kept_paths, keep_mask, scores


def plot_density(
    grid: np.ndarray,
    w_bins: np.ndarray,
    D: np.ndarray,
    stats: tuple[np.ndarray, ...],
    out_path: Path,
    grid_label: str,
    wmin: float,
    wmax: float,
    use_log: bool,
    reverse_cmap: bool,
    add_colorbar: bool,
    show_legend: bool,
    x_log10: bool,
    dpi: int,
    log_vmin: float | None,
) -> None:
    grid_edges = compute_edges(grid)
    base_cmap = plt.get_cmap("Blues")
    cmap = base_cmap.reversed() if reverse_cmap else base_cmap

    norm = None
    if use_log:
        positive = D[D > 0]
        if positive.size > 0:
            vmin = log_vmin if log_vmin is not None else positive.min()
            norm = LogNorm(vmin=vmin, vmax=positive.max())
        else:
            print("No positive density values; using linear scale.")

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(
        grid_edges,
        w_bins,
        D.T,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )

    if add_colorbar:
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(f"p(w | {grid_label})")

    mean, q16, q84, q025, q975, q005, q995 = stats
    c68 = base_cmap(0.85)
    c95 = base_cmap(0.65)
    c99 = base_cmap(0.45)
    ax.plot(grid, mean, color="white", lw=1.6, label="mean")
    ax.plot(grid, q16, color=c68, ls="--", lw=1.1, label="68%")
    ax.plot(grid, q84, color=c68, ls="--", lw=1.1)
    ax.plot(grid, q025, color=c95, ls=":", lw=1.1, label="95%")
    ax.plot(grid, q975, color=c95, ls=":", lw=1.1)
    ax.plot(grid, q005, color=c99, ls="-.", lw=1.1, label="99%")
    ax.plot(grid, q995, color=c99, ls="-.", lw=1.1)

    if x_log10:
        ax.set_xscale("log", base=10)
        ax.set_xlabel(f"{grid_label} (log10)")
    else:
        ax.set_xlabel(grid_label)
    ax.set_ylabel("w")
    ax.set_ylim(wmin, wmax)
    ax.set_xlim(grid.min(), grid.max())

    if show_legend:
        ax.legend(loc="best", frameon=True, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_quantile_file(
    out_path: Path,
    grid: np.ndarray,
    stats: tuple[np.ndarray, ...],
    grid_label: str,
) -> None:
    mean, q16, q84, q025, q975, q005, q995 = stats
    data = np.column_stack([grid, mean, q16, q84, q025, q975, q005, q995])
    header = f"{grid_label},mean,q16,q84,q025,q975,q005,q995"
    np.savetxt(out_path, data, delimiter=",", header=header)


def report_saved(path: Path, label: str) -> None:
    if not path.exists():
        print(f"WARNING: {label} not found after save: {path}")
        return
    size = path.stat().st_size
    if size == 0:
        print(f"WARNING: {label} saved but file is empty: {path}")


def main() -> int:
    folder_paths = normalize_folders(folder, folders)
    paths = collect_paths(folder_paths, pattern)
    if not paths:
        folder_str = ", ".join(str(p) for p in folder_paths)
        print(f"No files matched: {folder_str} / {pattern}")
        return 1

    W, grid, z_grid, skip, good_paths = load_wde_samples(paths, use_a=use_a)
    total = len(paths)
    kept = 0 if W is None else W.shape[0]
    print(f"Total files: {total}, kept: {kept}")
    for key, val in skip.items():
        if val:
            print(f"Skipped {key}: {val}")

    if W is None or grid is None or z_grid is None or kept == 0:
        print("No valid samples found; aborting.")
        return 1

    out_dir = Path(out_folder).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()
    print(f"Output directory: {out_dir}")

    weights = np.ones(W.shape[0], dtype=float)
    csv_paths = normalize_csv_paths(weight_csv)
    if csv_paths:
        path_map, base_map, weight_stats = load_weight_map(csv_paths, weight_col)
        weights, weight_keep_mask, match_stats = match_weights(
            good_paths, path_map, base_map, weight_key_mode, missing_weight
        )
        print(
            "Weight map: "
            f"rows={weight_stats['rows']} missing_files={weight_stats['missing_file']} "
            f"bad_cols={weight_stats['bad_cols']} bad_weight={weight_stats['bad_weight']} "
            f"duplicate_key={weight_stats['duplicate_key']} ambiguous_base={weight_stats['ambiguous_base']}"
        )
        print(
            "Weight match: "
            f"matched_path={match_stats['matched_path']} "
            f"matched_basename={match_stats['matched_basename']} "
            f"missing={match_stats['missing']} "
            f"ambiguous_basename={match_stats['ambiguous_basename']} "
            f"bad_weight={match_stats['bad_weight']}"
        )
        if not np.all(weight_keep_mask):
            W = W[weight_keep_mask]
            good_paths = [p for p, keep in zip(good_paths, weight_keep_mask) if keep]
            weights = weights[weight_keep_mask]
            print(f"Samples kept after weight filtering: {W.shape[0]}")
        if W.size == 0:
            print("No samples left after weight filtering; aborting.")
            return 1

    grid, z_grid, W = apply_z_range(grid, z_grid, W, zmin=zmin, zmax=zmax)
    if W.size == 0 or grid.size == 0:
        print("No bins left after z-range selection; aborting.")
        return 1
    if x_log10 and np.any(grid <= 0):
        print("ERROR: x_log10 requires positive grid values; set zmin>0 or use_a=True.")
        return 1

    all_good_paths = list(good_paths)
    W, weights, good_paths, keep_mask, _ = apply_trimming(
        W, weights, all_good_paths, trim_frac=trim_frac, trim_quantile=trim_quantile
    )
    if W.size == 0:
        print("No samples left after trimming; aborting.")
        return 1
    if write_trim_report and np.any(~keep_mask):
        trim_report = out_dir / "trimmed_samples.txt"
        with open(trim_report, "w", encoding="utf-8") as handle:
            handle.write("path\n")
            for path, keep in zip(all_good_paths, keep_mask):
                if not keep:
                    handle.write(f"{path}\n")
        print(f"Wrote trimmed samples list: {trim_report}")

    w_bins = np.linspace(wmin, wmax, n_wbin + 1)
    D = build_density(W, w_bins, weights)
    D = smooth_density(D, smooth_sigma)
    stats = compute_stats(W, weights)

    grid_label = "a" if use_a else "z"
    out_png = out_dir / f"wDE_density_{grid_label}.png"

    plot_density(
        grid=grid,
        w_bins=w_bins,
        D=D,
        stats=stats,
        out_path=out_png,
        grid_label=grid_label,
        wmin=wmin,
        wmax=wmax,
        use_log=use_log,
        reverse_cmap=reverse_cmap,
        add_colorbar=add_colorbar,
        show_legend=show_legend,
        x_log10=x_log10,
        dpi=dpi,
        log_vmin=log_vmin,
    )
    report_saved(out_png, "PNG")

    if write_quantiles:
        out_csv = out_dir / f"wDE_quantiles_{grid_label}.csv"
        write_quantile_file(out_csv, grid, stats, grid_label)
        report_saved(out_csv, "CSV")
        print(f"Saved: {out_csv}")

    print(f"Saved: {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
