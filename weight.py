#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np

# Parameters
folder = Path("/Users/dcz/data/Horndeski_samples/Horndeski_samples_py_onlybackground_a00_1")
pattern = "Horndeski_sample_*.dat"
out_csv = Path("/Users/dcz/data/weights/Horndeski_samples_py_onlybackground_a00_1.csv")

# Constants
C_KMS = 3e5
H0_OBS = 73.8
SIGMA_MILD = 4.8
Z_TOL = 1e-10

BAO_Z = np.array([0.106, 0.275, 0.35, 0.57], dtype=float)
BAO_X = np.array([0.336, 0.1390], dtype=float)
BAO_X_SIGMA = np.array([0.015, 0.0037], dtype=float)
BAO_Y_DIRECT = np.array([8.88, 13.67], dtype=float)
BAO_Y_DIRECT_SIGMA = np.array([0.17, 0.22], dtype=float)
BAO_MILD_FACTOR = 2.0

BAO_Y = np.concatenate((1.0 / BAO_X, BAO_Y_DIRECT))
BAO_SIGMA = np.concatenate(
    (BAO_X_SIGMA / (BAO_X ** 2), BAO_Y_DIRECT_SIGMA)
)
BAO_SIGMA_MILD = BAO_SIGMA * BAO_MILD_FACTOR
BAO_W = 1.0 / (BAO_SIGMA_MILD ** 2)
BAO_Z_MAX = float(BAO_Z.max())


def load_data(path: Path):
    try:
        data = np.loadtxt(path, comments="#")
    except ValueError:
        with path.open("r", encoding="utf-8") as handle:
            has_data = any(
                line.strip() and not line.lstrip().startswith("#")
                for line in handle
            )
        if not has_data:
            return None, "empty"
        return None, "bad_dim"

    if data.size == 0:
        return None, "empty"
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim != 2:
        return None, "bad_dim"
    if data.shape[1] <= 3:
        return None, "cols"
    if not np.isfinite(data).all():
        return None, "nonfinite"
    return data, None


def find_z0_row(data: np.ndarray):
    z_vals = data[:, 2]
    idxs = np.where(np.isclose(z_vals, 0.0, atol=Z_TOL, rtol=0.0))[0]
    if idxs.size:
        return int(idxs[0]), True
    return 0, False


def compute_bao(data: np.ndarray):
    z = data[:, 2]
    a = data[:, 1]
    h2 = data[:, 3]
    order = np.argsort(z)
    z = z[order]
    a = a[order]
    h2 = h2[order]

    if z[-1] < BAO_Z_MAX - Z_TOL:
        return None, "bao_zmax"

    h_phys = C_KMS * np.sqrt(h2) / a
    if not np.isfinite(h_phys).all() or np.any(h_phys <= 0):
        return None, "bao_nonfinite"

    dm = np.zeros_like(z)
    inv_h = C_KMS / h_phys
    dz = np.diff(z)
    dm[1:] = np.cumsum(0.5 * (inv_h[1:] + inv_h[:-1]) * dz)

    dm_k = np.interp(BAO_Z, z, dm)
    h_k = np.interp(BAO_Z, z, h_phys)
    if (
        not np.isfinite(dm_k).all()
        or not np.isfinite(h_k).all()
        or np.any(h_k <= 0)
    ):
        return None, "bao_nonfinite"

    dv = (dm_k ** 2 * (C_KMS * BAO_Z) / h_k) ** (1.0 / 3.0)
    if not np.isfinite(dv).all():
        return None, "bao_nonfinite"

    denom = np.sum(BAO_W * dv ** 2)
    if not np.isfinite(denom) or denom <= 0:
        return None, "bao_alpha"

    alpha_star = np.sum(BAO_W * dv * BAO_Y) / denom
    if not np.isfinite(alpha_star) or alpha_star == 0:
        return None, "bao_alpha"

    y_th = alpha_star * dv
    chi2_mild = np.sum(BAO_W * (y_th - BAO_Y) ** 2)
    if not np.isfinite(chi2_mild):
        return None, "bao_nonfinite"

    w_bao = float(np.exp(-0.5 * chi2_mild))
    r_s_star = 1.0 / alpha_star
    return (chi2_mild, w_bao, alpha_star, r_s_star), None


def main():
    files = sorted(folder.glob(pattern))
    total = len(files)
    results = []
    r_s_vals = []
    skip_counts = {
        "empty": 0,
        "bad_dim": 0,
        "cols": 0,
        "nonfinite": 0,
        "bao_zmax": 0,
        "bao_nonfinite": 0,
        "bao_alpha": 0,
    }
    missing_z0 = 0

    for path in files:
        data, reason = load_data(path)
        if reason:
            skip_counts[reason] += 1
            continue

        row_idx, has_z0 = find_z0_row(data)
        if not has_z0:
            missing_z0 += 1

        h2 = float(data[row_idx, 3])
        h0_th = C_KMS * np.sqrt(h2)
        if not np.isfinite(h0_th):
            skip_counts["nonfinite"] += 1
            continue

        chi2_mild = ((h0_th - H0_OBS) / SIGMA_MILD) ** 2
        w_h0 = float(np.exp(-0.5 * chi2_mild))

        bao_out, bao_reason = compute_bao(data)
        if bao_reason:
            skip_counts[bao_reason] += 1
            continue

        chi2_bao_mild, w_bao, alpha_star, r_s_star = bao_out
        w_total = w_h0 * w_bao
        results.append(
            (
                str(path),
                h0_th,
                chi2_mild,
                w_h0,
                chi2_bao_mild,
                w_bao,
                alpha_star,
                r_s_star,
                w_total,
            )
        )
        r_s_vals.append(r_s_star)

    if out_csv.parent != Path("."):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file",
                "H0_th",
                "chi2_H0_mild",
                "w_H0",
                "chi2_BAO_mild",
                "w_BAO",
                "alpha_star",
                "r_s_star",
                "w_total",
            ]
        )
        writer.writerows(results)

    success = len(results)
    skipped = total - success
    print(f"Total files: {total}")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print("Skip reasons:")
    for key in [
        "empty",
        "bad_dim",
        "cols",
        "nonfinite",
        "bao_zmax",
        "bao_nonfinite",
        "bao_alpha",
    ]:
        print(f"  {key}: {skip_counts[key]}")
    if missing_z0:
        print(
            f"Warning: z=0 row not found in {missing_z0} files; used first row."
        )

    if success:
        h0_vals = np.array([row[1] for row in results], dtype=float)
        print(f"H0_th mean: {h0_vals.mean():.6g}")
        print(f"H0_th median: {np.median(h0_vals):.6g}")
        print(f"H0_th std: {h0_vals.std(ddof=0):.6g}")

        r_s_arr = np.array(r_s_vals, dtype=float)
        if r_s_arr.size:
            p16, p50, p84 = np.percentile(r_s_arr, [16, 50, 84])
            print(f"r_s_star median: {p50:.6g}")
            print(f"r_s_star 16-84%: {p16:.6g} - {p84:.6g}")
    else:
        print("No successful files to compute H0 statistics.")


if __name__ == "__main__":
    main()
