#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np

# Parameters
folder = Path("/Users/dcz/data/Horndeski_samples/Horndeski_samples_py_onlybackground_a00_2")
pattern = "Horndeski_sample_*.dat"
out_csv = Path("/Users/dcz/data/weights/Horndeski_samples_py_onlybackground_a00_2.csv")
sn_mu_table_path = Path("/Users/dcz/data/SCPUnion2.1_mu_vs_z.txt")
sn_cov_path = Path("/Users/dcz/data/SCPUnion2.1_covmat_sys.txt")
use_sn_cov_systematics = True
sn_use_diag_only = False
mild_factor = 4.0

# Constants
C_KMS = 299792.458
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


def load_sn_inputs():
    sn_numeric = np.loadtxt(
        sn_mu_table_path, comments="#", usecols=(1, 2, 3)
    )
    if sn_numeric.ndim == 1:
        sn_numeric = sn_numeric.reshape(1, -1)

    z_sn = sn_numeric[:, 0]
    mu_obs = sn_numeric[:, 1]
    mu_err = sn_numeric[:, 2]
    n_sn = z_sn.size

    if sn_use_diag_only:
        if not np.isfinite(mu_err).all():
            raise ValueError("SN mu_err contains non-finite values.")
        cov = np.diag(mu_err ** 2)
        return z_sn, mu_obs, cov

    if not use_sn_cov_systematics and "sys" in sn_cov_path.name.lower():
        raise ValueError(
            "use_sn_cov_systematics=False but sn_cov_path looks like a "
            "systematics file; set sn_cov_path to a no-systematics "
            "covariance file or enable sn_use_diag_only."
        )

    cov_raw = np.loadtxt(sn_cov_path, comments="#")
    if cov_raw.ndim == 1:
        if cov_raw.size != n_sn * n_sn:
            raise ValueError("SN covariance size does not match N^2.")
        cov = cov_raw.reshape((n_sn, n_sn))
    elif cov_raw.ndim == 2:
        cov = cov_raw
    else:
        raise ValueError("SN covariance has unexpected dimensions.")

    if cov.shape != (n_sn, n_sn):
        raise ValueError("SN covariance shape does not match data length.")
    if not np.isfinite(cov).all():
        raise ValueError("SN covariance contains non-finite values.")
    return z_sn, mu_obs, cov


def compute_sn(
    data: np.ndarray, z_sn: np.ndarray, mu_obs: np.ndarray, cov: np.ndarray
):
    z = data[:, 2]
    a = data[:, 1]
    h2 = data[:, 3]
    order = np.argsort(z)
    z = z[order]
    a = a[order]
    h2 = h2[order]

    z_sn_max = float(np.max(z_sn))
    if z[-1] < z_sn_max - Z_TOL:
        return None, "sn_zmax"

    h_phys = C_KMS * np.sqrt(h2) / a
    if not np.isfinite(h_phys).all() or np.any(h_phys <= 0):
        return None, "sn_nonfinite"

    dm = np.zeros_like(z)
    inv_h = C_KMS / h_phys
    dz = np.diff(z)
    dm[1:] = np.cumsum(0.5 * (inv_h[1:] + inv_h[:-1]) * dz)

    dm_sn = np.interp(z_sn, z, dm)
    d_l = (1.0 + z_sn) * dm_sn
    if not np.isfinite(d_l).all() or np.any(d_l <= 0):
        return None, "sn_nonfinite"

    mu0 = 5.0 * np.log10(d_l) + 25.0
    if not np.isfinite(mu0).all():
        return None, "sn_nonfinite"

    r = mu0 - mu_obs
    ones = np.ones_like(r)
    try:
        v_r = np.linalg.solve(cov, r)
        v_1 = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError:
        return None, "sn_solve"

    A = r @ v_r
    B = ones @ v_r
    C_s = ones @ v_1
    if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C_s):
        return None, "sn_nonfinite"
    if C_s == 0:
        return None, "sn_solve"

    m_star = -B / C_s
    chi2_min = A - (B ** 2) / C_s
    if not np.isfinite(chi2_min):
        return None, "sn_nonfinite"

    chi2_mild = chi2_min / mild_factor
    w_sn = float(np.exp(-0.5 * chi2_mild))
    return (chi2_min, chi2_mild, w_sn, m_star), None


def main():
    files = sorted(folder.glob(pattern))
    total = len(files)
    results = []
    r_s_vals = []
    w_sn_vals = []
    m_star_vals = []
    skip_counts = {
        "empty": 0,
        "bad_dim": 0,
        "cols": 0,
        "nonfinite": 0,
        "bao_zmax": 0,
        "bao_nonfinite": 0,
        "bao_alpha": 0,
        "sn_zmax": 0,
        "sn_nonfinite": 0,
        "sn_solve": 0,
    }
    missing_z0 = 0
    z_sn, mu_obs, sn_cov = load_sn_inputs()

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

        sn_out, sn_reason = compute_sn(data, z_sn, mu_obs, sn_cov)
        if sn_reason:
            skip_counts[sn_reason] += 1
            continue

        chi2_sn_min, chi2_sn_mild, w_sn, m_star = sn_out
        w_total *= w_sn
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
                chi2_sn_min,
                chi2_sn_mild,
                w_sn,
                m_star,
                w_total,
            )
        )
        r_s_vals.append(r_s_star)
        w_sn_vals.append(w_sn)
        m_star_vals.append(m_star)

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
                "chi2_SN_min",
                "chi2_SN_mild",
                "w_SN",
                "M_star",
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
        "sn_zmax",
        "sn_nonfinite",
        "sn_solve",
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

        w_sn_arr = np.array(w_sn_vals, dtype=float)
        m_star_arr = np.array(m_star_vals, dtype=float)
        if w_sn_arr.size:
            w_p16, w_p50, w_p84 = np.percentile(w_sn_arr, [16, 50, 84])
            print(f"w_SN mean: {w_sn_arr.mean():.6g}")
            print(f"w_SN median: {w_p50:.6g}")
            print(f"w_SN 16-84%: {w_p16:.6g} - {w_p84:.6g}")
        if m_star_arr.size:
            m_p16, m_p50, m_p84 = np.percentile(m_star_arr, [16, 50, 84])
            print(f"M_star mean: {m_star_arr.mean():.6g}")
            print(f"M_star median: {m_p50:.6g}")
            print(f"M_star 16-84%: {m_p16:.6g} - {m_p84:.6g}")
    else:
        print("No successful files to compute H0 statistics.")


if __name__ == "__main__":
    main()
