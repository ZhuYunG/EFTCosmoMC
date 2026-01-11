#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np
try:
    from scipy.optimize import minimize, minimize_scalar
except ImportError as exc:
    minimize = None
    minimize_scalar = None
    _SCIPY_IMPORT_ERROR = exc


DEFAULT_CORR_PATH = (
    "/Volumes/My Passport/prior/Horndeski_samples_piror_max_wider_folder5/omegade_correlation.txt"
)
DEFAULT_BINS_PATH = (
    "/Volumes/My Passport/prior/Horndeski_samples_piror_max_wider_folder5/omegade_prior_bins.txt"
)
DEFAULT_COV_PATH = (
    "/Volumes/My Passport/prior/Horndeski_samples_piror_max_wider_folder5/omegade_covariance.txt"
)


def load_correlation_matrix(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2:
        raise ValueError(f"Correlation matrix must be 2D, got shape {data.shape}")
    return data


def load_covariance_matrix(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2:
        raise ValueError(f"Covariance matrix must be 2D, got shape {data.shape}")
    return data


def load_bins(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("Bins file must have at least two columns: a and z.")
    a = data[:, 0]
    z = data[:, 1]
    return a, z


def report_checks(a, z, C):
    n = a.shape[0]
    print(f"N = {n}")
    print(f"a shape = {a.shape}, z shape = {z.shape}, C shape = {C.shape}")
    if C.shape != (n, n):
        raise ValueError(f"Shape mismatch: C is {C.shape}, a is {a.shape}")
    if z.shape != (n,):
        raise ValueError(f"Shape mismatch: z is {z.shape}, a is {a.shape}")

    if not np.isfinite(a).all():
        idx = np.argwhere(~np.isfinite(a)).ravel()
        print(f"Non-finite values in a: count={idx.size}, indices (0-based)={idx[:10]}")
        raise ValueError("Non-finite values in a; aborting.")
    if not np.isfinite(z).all():
        idx = np.argwhere(~np.isfinite(z)).ravel()
        print(f"Non-finite values in z: count={idx.size}, indices (0-based)={idx[:10]}")
        raise ValueError("Non-finite values in z; aborting.")

    nonfinite = ~np.isfinite(C)
    if nonfinite.any():
        idx = np.argwhere(nonfinite)
        print(
            f"Non-finite values in C: count={idx.shape[0]}, "
            f"first indices (0-based)={idx[:10].tolist()}"
        )
    else:
        print("C has no non-finite values.")

    mask = np.isfinite(C) & np.isfinite(C.T)
    if mask.any():
        sym_err = np.max(np.abs(C - C.T)[mask])
    else:
        sym_err = float("nan")
    print(f"sym_err = {sym_err:.6e}")

    diag = np.diag(C)
    diag_mask = np.isfinite(diag)
    if diag_mask.any():
        diag_err = np.max(np.abs(diag[diag_mask] - 1.0))
    else:
        diag_err = float("nan")
    print(f"diag_err = {diag_err:.6e}")


def report_bins_checks(a, z):
    n = a.shape[0]
    print(f"N = {n}")
    print(f"a shape = {a.shape}, z shape = {z.shape}")
    if z.shape != (n,):
        raise ValueError(f"Shape mismatch: z is {z.shape}, a is {a.shape}")
    if not np.isfinite(a).all():
        idx = np.argwhere(~np.isfinite(a)).ravel()
        print(f"Non-finite values in a: count={idx.size}, indices (0-based)={idx[:10]}")
        raise ValueError("Non-finite values in a; aborting.")
    if not np.isfinite(z).all():
        idx = np.argwhere(~np.isfinite(z)).ravel()
        print(f"Non-finite values in z: count={idx.size}, indices (0-based)={idx[:10]}")
        raise ValueError("Non-finite values in z; aborting.")


def build_samples(x, C):
    n = x.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)
    a_i = x[i_idx]
    a_j = x[j_idx]
    y = C[i_idx, j_idx]
    mask = np.isfinite(a_i) & np.isfinite(a_j) & np.isfinite(y)
    d = np.abs(a_i[mask] - a_j[mask])
    y = y[mask]
    return d, y


def build_diag_samples(x, cov):
    diag = np.diag(cov)
    mask = np.isfinite(x) & np.isfinite(diag)
    return x[mask], diag[mask]


def model_c16(dist, xi):
    return 1.0 / (1.0 + (dist / xi) ** 2)


def model_c17(dist, xi, n):
    return np.exp(-((dist / xi) ** n))


def sse_for_model(dist, obs, xi, model_fn):
    pred = model_fn(dist, xi)
    resid = pred - obs
    return float(np.sum(resid * resid))


def sse_for_c17(dist, obs, xi, n):
    pred = model_c17(dist, xi, n)
    resid = pred - obs
    return float(np.sum(resid * resid))


def guess_xi(dist, obs, target):
    idx = int(np.argmin(np.abs(obs - target)))
    xi0 = float(dist[idx])
    if not np.isfinite(xi0) or xi0 <= 0:
        xi0 = float(np.median(dist))
    return xi0


def coarse_search(dist, obs, model_fn, xi0, min_xi, max_xi, num=80):
    lo = max(min_xi, xi0 / 100.0)
    hi = min(max_xi, xi0 * 100.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= 0 or lo >= hi:
        lo, hi = min_xi, max_xi
    grid = np.logspace(math.log10(lo), math.log10(hi), num=num)
    sse_vals = np.array([sse_for_model(dist, obs, xi, model_fn) for xi in grid])
    i_best = int(np.argmin(sse_vals))
    xi_best = float(grid[i_best])
    return xi_best, float(sse_vals[i_best]), lo, hi


def golden_search_log(dist, obs, model_fn, lo, hi, tol=1e-8, max_iter=200):
    if lo <= 0 or hi <= 0 or lo >= hi:
        raise ValueError("Invalid bracket for golden search.")
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    log_lo = math.log(lo)
    log_hi = math.log(hi)
    c = log_hi - (log_hi - log_lo) * inv_phi
    d = log_lo + (log_hi - log_lo) * inv_phi
    fc = sse_for_model(dist, obs, math.exp(c), model_fn)
    fd = sse_for_model(dist, obs, math.exp(d), model_fn)

    iters = 0
    while (log_hi - log_lo) > tol and iters < max_iter:
        if fc < fd:
            log_hi = d
            d = c
            fd = fc
            c = log_hi - (log_hi - log_lo) * inv_phi
            fc = sse_for_model(dist, obs, math.exp(c), model_fn)
        else:
            log_lo = c
            c = d
            fc = fd
            d = log_lo + (log_hi - log_lo) * inv_phi
            fd = sse_for_model(dist, obs, math.exp(d), model_fn)
        iters += 1

    xi_best = math.exp((log_lo + log_hi) / 2.0)
    sse_best = sse_for_model(dist, obs, xi_best, model_fn)
    success = iters < max_iter
    return xi_best, sse_best, iters, success


def fit_model_c16(dist, obs, model_fn, target, name):
    min_d = float(np.min(dist))
    max_d = float(np.max(dist))
    min_xi = max(min_d * 1e-3, 1e-12)
    max_xi = max_d * 1e3

    xi0 = guess_xi(dist, obs, target)
    xi_seed, sse_seed, lo, hi = coarse_search(dist, obs, model_fn, xi0, min_xi, max_xi)
    lo_ref = max(min_xi, xi_seed / 10.0)
    hi_ref = min(max_xi, xi_seed * 10.0)
    if lo_ref >= hi_ref:
        lo_ref, hi_ref = lo, hi

    if minimize_scalar is None:
        print(f"[{name}] SciPy not available: {_SCIPY_IMPORT_ERROR}")
        xi_opt, sse_opt, iters, success = golden_search_log(
            dist, obs, model_fn, lo_ref, hi_ref
        )
    else:
        result = minimize_scalar(
            lambda xi: sse_for_model(dist, obs, xi, model_fn),
            bounds=(lo_ref, hi_ref),
            method="bounded",
            options={"xatol": 1e-8, "maxiter": 500},
        )
        xi_opt = float(result.x)
        sse_opt = float(result.fun)
        iters = int(result.nit)
        success = bool(result.success)
    print(f"[{name}] initial guess xi0 = {xi0:.6e}")
    print(f"[{name}] search bracket = [{lo_ref:.6e}, {hi_ref:.6e}]")
    print(f"[{name}] optimizer: iters={iters}, success={success}, SSE={sse_opt:.6e}")
    return xi_opt, sse_opt, iters, success


def fit_model_c17(dist, obs, target, name):
    if minimize is None:
        raise RuntimeError(f"SciPy not available: {_SCIPY_IMPORT_ERROR}")

    min_d = float(np.min(dist))
    max_d = float(np.max(dist))
    min_xi = max(min_d * 1e-3, 1e-12)
    max_xi = max_d * 1e3
    min_n, max_n = 0.1, 10.0

    xi0 = guess_xi(dist, obs, target)
    n0 = 2.0

    bounds = [(min_xi, max_xi), (min_n, max_n)]
    result = minimize(
        lambda x: sse_for_c17(dist, obs, x[0], x[1]),
        x0=np.array([xi0, n0], dtype=float),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )

    xi_opt = float(result.x[0])
    n_opt = float(result.x[1])
    sse_opt = float(result.fun)
    iters = int(getattr(result, "nit", 0))
    success = bool(result.success)

    print(f"[{name}] initial guess xi0 = {xi0:.6e}, n0 = {n0:.6e}")
    print(
        f"[{name}] bounds xi in [{min_xi:.6e}, {max_xi:.6e}], "
        f"n in [{min_n:.6e}, {max_n:.6e}]"
    )
    print(f"[{name}] optimizer: iters={iters}, success={success}, SSE={sse_opt:.6e}")
    return xi_opt, n_opt, sse_opt, iters, success


def build_prediction_and_residual(x, C, model_fn, xi):
    dist_mat = np.abs(x[:, None] - x[None, :])
    chat = model_fn(dist_mat, xi)
    resid = chat - C
    return chat, resid


def select_distance_array(a, z, d_mode):
    if d_mode == "a":
        return a, "|a_i - a_j|"
    if d_mode == "z":
        return z, "|z_i - z_j|"
    if d_mode == "ln_a":
        return np.log(a), "|ln a_i - ln a_j|"
    raise ValueError(f"Unknown d_mode: {d_mode}")


def select_x_array(a, z, x_mode):
    if x_mode == "a":
        return a, "a", 1.0
    if x_mode == "z":
        return z, "z", 0.0
    if x_mode == "ln_a":
        return np.log(a), "ln a", 0.0
    raise ValueError(f"Unknown x_mode: {x_mode}")


def validate_ln_a(a, label):
    if np.any(a <= 0):
        idx = np.argwhere(a <= 0).ravel()
        print(
            f"Non-positive values in a for {label}: count={idx.size}, "
            f"indices (0-based)={idx[:10]}"
        )
        raise ValueError("Non-positive values in a; cannot take ln(a).")


def model_1d_taylor_series(x, coeffs, x0):
    dx = x - x0
    out = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        out += c * np.power(dx, i)
    return out


def model_1d_exp_series(x, coeffs, exponents, x0):
    if len(coeffs) != len(exponents) + 1:
        raise ValueError("exp series expects len(coeffs) = len(exponents) + 1.")
    dx = x - x0
    out = np.zeros_like(x, dtype=float)
    out += coeffs[0]
    for c, a in zip(coeffs[1:], exponents):
        out += c * np.exp(a * dx)
    return out


def model_1d_power_series(x, coeffs, exponents, x0):
    if len(coeffs) != len(exponents) + 1:
        raise ValueError("power series expects len(coeffs) = len(exponents) + 1.")
    base = np.abs(x - x0)
    out = np.zeros_like(x, dtype=float)
    out += coeffs[0]
    for c, a in zip(coeffs[1:], exponents):
        out += c * np.power(base, a)
    return out


def model_1d_taylor(x, alpha, beta, x0):
    return model_1d_taylor_series(x, [alpha, beta], x0)


def model_1d_exp(x, alpha, beta, gamma, x0):
    return model_1d_exp_series(x, [alpha, beta], [gamma], x0)


def model_1d_power(x, alpha, beta, gamma, x0):
    return model_1d_power_series(x, [alpha, beta], [gamma], x0)


def sse_1d(pred, obs):
    if pred is None or (not np.isfinite(pred).all()):
        return float("inf")
    resid = pred - obs
    return float(np.sum(resid * resid))


def rmse_1d(pred, obs):
    resid = pred - obs
    return math.sqrt(float(np.mean(resid * resid)))


def fit_1d_taylor(x, y, x0, fit_x0, order=1):
    if order < 1:
        raise ValueError("Taylor order must be >= 1.")
    if not fit_x0:
        dx = x - x0
        A = np.column_stack([np.power(dx, i) for i in range(order + 1)])
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        coeffs = [float(p) for p in params]
        pred = model_1d_taylor_series(x, coeffs, x0)
        sse = sse_1d(pred, y)
        rmse = rmse_1d(pred, y)
        return {
            "name": "taylor",
            "terms": order,
            "coeffs": coeffs,
            "exponents": None,
            "alpha": coeffs[0],
            "beta": coeffs[1] if order >= 1 else None,
            "gamma": None,
            "x0": x0,
            "sse": sse,
            "rmse": rmse,
            "iters": 1,
            "success": True,
        }

    if minimize is None:
        raise RuntimeError(f"SciPy not available: {_SCIPY_IMPORT_ERROR}")

    x0_bounds = (np.min(x), np.max(x))
    x0_init = float(np.clip(x0, *x0_bounds))
    dx = x - x0_init
    A = np.column_stack([np.power(dx, i) for i in range(order + 1)])
    init_coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    def obj(p):
        coeffs = p[:-1]
        x0p = p[-1]
        pred = model_1d_taylor_series(x, coeffs, x0p)
        return sse_1d(pred, y)

    init = np.concatenate([init_coeffs, [x0_init]])
    bounds = [(None, None)] * (order + 1) + [x0_bounds]
    result = minimize(
        obj,
        x0=np.array(init, dtype=float),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )

    coeffs = [float(v) for v in result.x[:-1]]
    x0_opt = float(result.x[-1])
    pred = model_1d_taylor_series(x, coeffs, x0_opt)
    return {
        "name": "taylor",
        "terms": order,
        "coeffs": coeffs,
        "exponents": None,
        "alpha": coeffs[0],
        "beta": coeffs[1] if order >= 1 else None,
        "gamma": None,
        "x0": x0_opt,
        "sse": sse_1d(pred, y),
        "rmse": rmse_1d(pred, y),
        "iters": int(getattr(result, "nit", 0)),
        "success": bool(result.success),
    }


def fit_1d_exp(x, y, x0, fit_x0, terms=1):
    if terms < 1:
        raise ValueError("Exponential terms must be >= 1.")
    if minimize is None:
        raise RuntimeError(f"SciPy not available: {_SCIPY_IMPORT_ERROR}")

    x0_bounds = (np.min(x), np.max(x))
    x0_init = float(np.clip(x0, *x0_bounds))
    x0_param = x0_init if fit_x0 else x0
    coeff_bounds = [(None, None)] * (terms + 1)
    exp_bounds = [(-10.0, 10.0)] * terms
    bounds = coeff_bounds + exp_bounds
    if fit_x0:
        bounds.append(x0_bounds)

    def obj(p):
        coeffs = p[: terms + 1]
        exponents = p[terms + 1 : terms + 1 + terms]
        if fit_x0:
            x0p = p[-1]
        else:
            x0p = x0_param
        pred = model_1d_exp_series(x, coeffs, exponents, x0p)
        return sse_1d(pred, y)

    alpha_guess = float(np.median(y))
    beta_guess = float((np.max(y) - np.min(y)) or 1.0)
    coeffs_guess = [alpha_guess] + [beta_guess / terms] * terms
    if terms == 1:
        exponents_guess = [0.0]
    else:
        exponents_guess = np.linspace(-1.0, 1.0, terms).tolist()
    init = coeffs_guess + exponents_guess
    if fit_x0:
        init.append(x0_init)

    result = minimize(
        obj,
        x0=np.array(init, dtype=float),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )

    coeffs = [float(v) for v in result.x[: terms + 1]]
    exponents = [float(v) for v in result.x[terms + 1 : terms + 1 + terms]]
    x0_opt = float(result.x[-1]) if fit_x0 else x0_param
    pred = model_1d_exp_series(x, coeffs, exponents, x0_opt)
    return {
        "name": "exp",
        "terms": terms,
        "coeffs": coeffs,
        "exponents": exponents,
        "alpha": coeffs[0],
        "beta": coeffs[1] if terms >= 1 else None,
        "gamma": exponents[0] if terms >= 1 else None,
        "x0": x0_opt,
        "sse": sse_1d(pred, y),
        "rmse": rmse_1d(pred, y),
        "iters": int(getattr(result, "nit", 0)),
        "success": bool(result.success),
    }


def fit_1d_power(x, y, x0, fit_x0, terms=1):
    if terms < 1:
        raise ValueError("Power-law terms must be >= 1.")
    if minimize is None:
        raise RuntimeError(f"SciPy not available: {_SCIPY_IMPORT_ERROR}")

    min_x = float(np.min(x))
    max_x = float(np.max(x))
    x0_bounds = (min_x, max_x)
    x0_init = float(np.clip(x0, *x0_bounds))
    x0_param = x0_init if fit_x0 else x0
    coeff_bounds = [(None, None)] * (terms + 1)
    pow_bounds = [(1e-6, 10.0)] * terms
    bounds = coeff_bounds + pow_bounds
    if fit_x0:
        bounds.append(x0_bounds)

    def obj(p):
        coeffs = p[: terms + 1]
        exponents = p[terms + 1 : terms + 1 + terms]
        if fit_x0:
            x0p = p[-1]
        else:
            x0p = x0_param
        pred = model_1d_power_series(x, coeffs, exponents, x0p)
        return sse_1d(pred, y)

    alpha_guess = float(np.median(y))
    beta_guess = float((np.max(y) - np.min(y)) or 1.0)
    coeffs_guess = [alpha_guess] + [beta_guess / terms] * terms
    exponents_guess = [1.0] * terms
    init = coeffs_guess + exponents_guess
    if fit_x0:
        init.append(x0_init)

    result = minimize(
        obj,
        x0=np.array(init, dtype=float),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )

    coeffs = [float(v) for v in result.x[: terms + 1]]
    exponents = [float(v) for v in result.x[terms + 1 : terms + 1 + terms]]
    x0_opt = float(result.x[-1]) if fit_x0 else x0_param
    pred = model_1d_power_series(x, coeffs, exponents, x0_opt)
    return {
        "name": "power",
        "terms": terms,
        "coeffs": coeffs,
        "exponents": exponents,
        "alpha": coeffs[0],
        "beta": coeffs[1] if terms >= 1 else None,
        "gamma": exponents[0] if terms >= 1 else None,
        "x0": x0_opt,
        "sse": sse_1d(pred, y),
        "rmse": rmse_1d(pred, y),
        "iters": int(getattr(result, "nit", 0)),
        "success": bool(result.success),
    }


def format_1d_model(result):
    x0 = result["x0"]
    coeffs = result["coeffs"]
    exponents = result["exponents"] or []
    if result["name"] == "taylor":
        terms = []
        for i, c in enumerate(coeffs):
            if i == 0:
                terms.append(f"{c:.6e}")
            else:
                terms.append(f"{c:.6e} * (x - {x0:.6e})^{i}")
        return "taylor: C(x) = " + " + ".join(terms)
    if result["name"] == "exp":
        terms = [f"{coeffs[0]:.6e}"]
        for c, a in zip(coeffs[1:], exponents):
            terms.append(f"{c:.6e} * exp({a:.6e} * (x - {x0:.6e}))")
        return "exp: C(x) = " + " + ".join(terms)
    if result["name"] == "power":
        terms = [f"{coeffs[0]:.6e}"]
        for c, a in zip(coeffs[1:], exponents):
            terms.append(f"{c:.6e} * |x - {x0:.6e}|^{a:.6e}")
        return "power: C(x) = " + " + ".join(terms)
    return f"{result['name']}: unknown model"


def main():
    parser = argparse.ArgumentParser(description="Fit fixed n=2 correlation models.")
    parser.add_argument("--corr", default=DEFAULT_CORR_PATH, help="Path to C matrix.")
    parser.add_argument("--bins", default=DEFAULT_BINS_PATH, help="Path to bins file.")
    parser.add_argument(
        "--d-mode",
        default="a",
        choices=["a", "z", "ln_a"],
        help="Distance variable: a, z, or ln_a (default: a).",
    )
    parser.add_argument(
        "--cov",
        default=DEFAULT_COV_PATH,
        help="Path to covariance matrix for 1D diagonal fitting.",
    )
    parser.add_argument(
        "--x-mode",
        default="a",
        choices=["a", "z", "ln_a"],
        help="1D x variable for diagonal fit: a, z, or ln_a (default: a).",
    )
    parser.add_argument(
        "--taylor-order",
        type=int,
        default=1,
        help="Max power for Taylor 1D fit (default: 1).",
    )
    parser.add_argument(
        "--exp-terms",
        type=int,
        default=1,
        help="Number of exponential terms for 1D fit (default: 1).",
    )
    parser.add_argument(
        "--power-terms",
        type=int,
        default=1,
        help="Number of power-law terms for 1D fit (default: 1).",
    )
    parser.add_argument(
        "--fit-x0",
        action="store_true",
        help="Fit x0 for 1D models instead of fixing it to 0 (z) or 1 (a).",
    )
    parser.add_argument(
        "--fit-mode",
        default="both",
        choices=["both", "1d", "2d"],
        help="Run both fits or only 1d/2d (default: both).",
    )
    parser.add_argument("--out-dir", default=".", help="Output directory for matrices.")
    args = parser.parse_args()

    corr_path = Path(args.corr)
    bins_path = Path(args.bins)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory = {out_dir.resolve()}")

    a, z = load_bins(bins_path)
    do_2d = args.fit_mode in ("both", "2d")
    do_1d = args.fit_mode in ("both", "1d")
    if do_1d:
        if args.taylor_order < 1:
            raise ValueError("Taylor order must be >= 1.")
        if args.exp_terms < 1:
            raise ValueError("Exponential terms must be >= 1.")
        if args.power_terms < 1:
            raise ValueError("Power-law terms must be >= 1.")

    if do_2d:
        C = load_correlation_matrix(corr_path)
        report_checks(a, z, C)
    else:
        report_bins_checks(a, z)

    if (do_2d and args.d_mode == "ln_a") or (do_1d and args.x_mode == "ln_a"):
        validate_ln_a(a, "ln_a mode")
    if do_2d:
        x, d_desc = select_distance_array(a, z, args.d_mode)

        dist, obs = build_samples(x, C)
        if dist.size == 0:
            raise ValueError("No valid samples after filtering upper triangle.")
        print(f"K (upper-triangle samples) = {dist.size}")
        print(f"Using d = {d_desc} for fitting.")

        xi16, sse16, _, _ = fit_model_c16(dist, obs, model_c16, target=0.5, name="C16")
        xi17, n17, sse17, _, _ = fit_model_c17(
            dist, obs, target=math.exp(-1.0), name="C17"
        )

        resid16 = model_c16(dist, xi16) - obs
        resid17 = model_c17(dist, xi17, n17) - obs
        rmse16 = math.sqrt(float(np.mean(resid16 * resid16)))
        rmse17 = math.sqrt(float(np.mean(resid17 * resid17)))

        print("\nModel forms with fitted xi:")
        print(f"C16(d) = 1 / (1 + (d/{xi16:.6e})^2 ), d = {d_desc}")
        print(f"C17(d) = exp( - (d/{xi17:.6e})^{n17:.6e} ), d = {d_desc}")
        print(f"C16 SSE = {sse16:.6e}, RMSE = {rmse16:.6e}")
        print(f"C17 SSE = {sse17:.6e}, RMSE = {rmse17:.6e}")

        chat16, r16 = build_prediction_and_residual(x, C, model_c16, xi16)
        dist_mat = np.abs(x[:, None] - x[None, :])
        chat17 = model_c17(dist_mat, xi17, n17)
        r17 = chat17 - C

        pred16_path = out_dir / "C16_prediction.txt"
        resid16_path = out_dir / "C16_residual.txt"
        pred17_path = out_dir / "C17_prediction.txt"
        resid17_path = out_dir / "C17_residual.txt"

        np.savetxt(pred16_path, chat16, fmt="%.18e")
        np.savetxt(resid16_path, r16, fmt="%.18e")
        np.savetxt(pred17_path, chat17, fmt="%.18e")
        np.savetxt(resid17_path, r17, fmt="%.18e")

        print("\nSaved matrices:")
        print(f"- {pred16_path}")
        print(f"- {resid16_path}")
        print(f"- {pred17_path}")
        print(f"- {resid17_path}")

        results = [
            {"name": "fixed-(16)", "xi": xi16, "n": 2.0, "sse": sse16, "rmse": rmse16},
            {"name": "free-(17)", "xi": xi17, "n": n17, "sse": sse17, "rmse": rmse17},
        ]
        best = min(results, key=lambda r: r["rmse"])

        print("\nSummary table (RMSE as residual):")
        print(f"{'model':<12} {'xi_*':>12} {'n_*':>12} {'RMSE':>12} {'SSE':>12}")
        for r in results:
            print(
                f"{r['name']:<12} {r['xi']:>12.6g} {r['n']:>12.6g} "
                f"{r['rmse']:>12.6g} {r['sse']:>12.6g}"
            )
        print(f"\nBest model by RMSE: {best['name']}")
    else:
        print("\nSkipping 2D fit (--fit-mode 1d).")

    if not do_1d:
        print("\nSkipping 1D diagonal fit (--fit-mode 2d).")
        return

    if not args.cov:
        raise ValueError("Covariance path is required for 1D fit.")

    cov_path = Path(args.cov)
    if not cov_path.exists():
        raise ValueError(f"Covariance path not found: {cov_path}")
    cov = load_covariance_matrix(cov_path)
    if cov.shape != (a.shape[0], a.shape[0]):
        raise ValueError(
            f"Covariance shape {cov.shape} does not match a length {a.shape[0]}."
        )

    x1, x_desc, x0_default = select_x_array(a, z, args.x_mode)
    x1, y1 = build_diag_samples(x1, cov)
    if x1.size == 0:
        raise ValueError("No valid diagonal samples for 1D fit.")

    print("\n1D diagonal covariance fit:")
    if args.fit_x0:
        print(f"Using x = {x_desc}, x0 is free")
    else:
        print(f"Using x = {x_desc}, x0 fixed = {x0_default:.6g}")

    results_1d = []
    results_1d.append(
        fit_1d_taylor(x1, y1, x0_default, args.fit_x0, order=args.taylor_order)
    )
    results_1d.append(
        fit_1d_exp(x1, y1, x0_default, args.fit_x0, terms=args.exp_terms)
    )
    results_1d.append(
        fit_1d_power(x1, y1, x0_default, args.fit_x0, terms=args.power_terms)
    )

    print("\n1D model forms with fitted parameters:")
    for r in results_1d:
        print(format_1d_model(r))

    print("\n1D summary table (RMSE as residual):")
    print(f"{'model':<10} {'terms':>5} {'x0':>12} {'RMSE':>12} {'SSE':>12}")
    for r in results_1d:
        print(
            f"{r['name']:<10} {r['terms']:>5d} {r['x0']:>12.6g} "
            f"{r['rmse']:>12.6g} {r['sse']:>12.6g}"
        )
    best_1d = min(results_1d, key=lambda r: r["rmse"])
    print(f"\nBest 1D model by RMSE: {best_1d['name']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
