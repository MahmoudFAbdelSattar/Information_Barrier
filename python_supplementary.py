
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Supplementary Material
#
# "The Information Barrier of Algebraic Wall Functions:
#  Why U and k Cannot Determine the Arcsinh-Transformed Velocity Variance"
#
# Submitted to Physics of Fluids (Standard Paper)
#
# Author: Mahmoud F. Abdel-Sattar
# Email:  m.f.abdel-sattar@azhar.edu.eg
# Affiliation: Department of Astronomy and Meteorology,
#              Faculty of Science, Al-Azhar University, Cairo, Egypt
#
# =============================================================================
#
# Methodological notes
# --------------------
# 1. DNS-parameterized Gaussian Monte Carlo surrogate; NOT direct-DNS
#    instantaneous velocity sampling.
# 2. Main anisotropy interval: A in [0,2].
#    Extended numerical sweep: A in [0,10].
# 3. Common random numbers are used across A in anisotropy sweeps.
#    This makes adjacent differences statistically paired.
# 4. The paired standard error is estimated using the influence function
#    of the sample variance.
# 5. Both pointwise and Bonferroni family-wise thresholds are reported.
# 6. Multi-seed confidence intervals quantify seed-to-seed variability only.
#    They exclude Gaussian model-form uncertainty.
# 7. A1, A2, A3 are sequential diagnostics, not independent causal effects.
#
# Requirements: numpy, scipy, matplotlib
# Usage: python supplementary.py
# =============================================================================

from __future__ import annotations

import os
import csv
import json
import math
import tarfile
import zipfile
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

RE_LIST = (180, 395, 590)

MC_SAMPLES = 200_000
MC_SEED = 42

ROBUST_SEEDS = (11, 29, 47, 71, 1001)
ROBUST_SAMPLES = 200_000

CONV_SEEDS = (11, 29, 47, 71, 101, 137, 173, 211)
CONV_SAMPLE_SIZES = (25_000, 100_000, 400_000, 1_600_000)
CONV_A_VALUES = (0.0, 0.5, 1.0, 2.0)

Y_PLUS_MIN = 0.5
Y_PLUS_MAX = 60.0

A_MIN = 0.0
A_MAX = 2.0
A_POINTS = 21

A_MAX_EXTENDED = 10.0
A_POINTS_EXTENDED = 1001

COVARIANCE_TOL = 1e-10

# Paired local-monotonicity diagnostic
PAIRED_DIAGNOSTIC_SEEDS = (29, 71)
PAIRED_DIAGNOSTIC_SAMPLES = ROBUST_SAMPLES
PAIRED_ALPHA = 0.05

plt.rc("font", family="serif", size=10)
plt.rc("mathtext", fontset="cm")
plt.rc("axes", linewidth=0.8)
plt.rc("lines", linewidth=1.2)


# =============================================================================
# §0 DATA PREPARATION
# =============================================================================

def _safe_extract_tar(tar_path: str, destination: str = ".") -> None:
    destination = os.path.abspath(destination)

    with tarfile.open(tar_path, "r:*") as tar:
        members = tar.getmembers()

        for member in members:
            target = os.path.abspath(
                os.path.join(destination, member.name)
            )

            if not target.startswith(destination + os.sep) and target != destination:
                raise RuntimeError(
                    f"Unsafe archive member: {member.name}"
                )

            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Links not permitted: {member.name}"
                )

            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    f"Unsupported archive member type: {member.name}"
                )

        try:
            tar.extractall(
                destination,
                members=members,
                filter="data"
            )
        except TypeError:
            # Older Python fallback: reject special files and links above.
            tar.extractall(destination, members=members)


def _safe_extract_zip(zip_path: str, destination: str = ".") -> None:
    destination = os.path.abspath(destination)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = os.path.abspath(
                os.path.join(destination, member.filename)
            )

            if not target.startswith(destination + os.sep) and target != destination:
                raise RuntimeError(
                    f"Unsafe archive member: {member.filename}"
                )

            # Reject symbolic links.
            unix_mode = (member.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise RuntimeError(
                    f"Links not permitted: {member.filename}"
                )

        zf.extractall(destination)


def prepare_data() -> None:
    expected = [f"chan{Re}" for Re in RE_LIST]

    if all(os.path.isdir(folder) for folder in expected):
        print("[prepare_data] DNS folders found locally.")
        return

    try:
        import google.colab  # noqa: F401
        in_colab = True
    except ImportError:
        in_colab = False

    if not in_colab:
        raise FileNotFoundError(
            "DNS folders not found. Required: chan180/, chan395/, chan590/.\n"
            "Each must contain a profiles/ subdirectory."
        )

    from google.colab import files

    print("[prepare_data] Please upload chandata.tar.gz, .tgz, or .zip.")
    uploaded = files.upload()

    for fname in uploaded.keys():
        print(f"[prepare_data] Processing: {fname}")

        try:
            if fname.endswith((".tar.gz", ".tgz")):
                _safe_extract_tar(fname, ".")
            elif fname.endswith(".zip"):
                _safe_extract_zip(fname, ".")
            else:
                print(f"[prepare_data] Unsupported format: {fname}")
                continue
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    missing = [
        folder for folder in expected
        if not os.path.isdir(folder)
    ]

    if missing:
        raise RuntimeError(
            f"Extraction incomplete. Missing: {missing}"
        )

    print("[prepare_data] DNS folders prepared.")


# =============================================================================
# §1 DNS DATA LOADING
# =============================================================================

def _load_profile_file(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    data = np.loadtxt(path, comments="#")

    if data.ndim != 2:
        raise ValueError(f"Expected 2D table: {path}")

    if data.shape[1] < 6:
        raise ValueError(f"Expected >=6 columns in {path}")

    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite values in {path}")

    return data


def load_dns(Re_tau: int) -> dict:
    folder = os.path.join(f"chan{Re_tau}", "profiles")

    rs = _load_profile_file(
        os.path.join(folder, f"chan{Re_tau}.reystress")
    )

    means = _load_profile_file(
        os.path.join(folder, f"chan{Re_tau}.means")
    )

    if rs.shape[0] != means.shape[0]:
        raise ValueError(
            f"Row-count mismatch for Re={Re_tau}"
        )

    if not np.allclose(
        rs[:, 1], means[:, 1],
        rtol=1e-8, atol=1e-10
    ):
        raise ValueError(
            f"Wall-normal mismatch for Re={Re_tau}"
        )

    mask = rs[:, 1] > 0.0

    return {
        "y_plus": rs[mask, 1],
        "uu": rs[mask, 2],
        "vv": rs[mask, 3],
        "ww": rs[mask, 4],
        "uv": rs[mask, 5],
        "U_plus": means[mask, 2],
    }


# =============================================================================
# §2 EXACT HYPERBOLIC IDENTITY
# =============================================================================

def rhs_identity(U: float, k: float) -> float:
    return 1.0 + 2.0 * U**2 + 4.0 * k


# =============================================================================
# §3 CLOSED-FORM CANDIDATES
# =============================================================================

def F1_linear_u(U: float, uu: float) -> float:
    return uu / (1.0 + U**2)


def F2_isotropic(U: float, k: float) -> float:
    return (2.0 * k / 3.0) / (1.0 + U**2)


def F3_small_fluct(U: float, k: float) -> float:
    return 2.0 * k / (1.0 + 2.0 * U**2)


def F4_closed(U: float, k: float) -> float:
    arg = 1.0 + 4.0 * k / (1.0 + 2.0 * U**2)

    if arg <= 0.0:
        return float("nan")

    return 0.5 * math.log(arg)


# =============================================================================
# §4 DNS-PARAMETERIZED GAUSSIAN MONTE CARLO
# =============================================================================

def _build_covariance(uu, vv, ww, uv=0.0) -> np.ndarray:
    cov = np.array(
        [
            [uu, uv, 0.0],
            [uv, vv, 0.0],
            [0.0, 0.0, ww],
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(cov)):
        raise ValueError(
            "Covariance contains non-finite entries."
        )

    eigs = np.linalg.eigvalsh(cov)
    scale = max(float(np.max(np.abs(eigs))), 1.0)
    tol = COVARIANCE_TOL * scale

    if eigs[0] < -tol:
        raise ValueError(
            f"Non-PSD covariance: min eigenvalue = {eigs[0]:.3e} "
            f"(tol = {tol:.3e})"
        )

    if eigs[0] < 0.0:
        warnings.warn(
            f"Covariance has a slightly negative eigenvalue "
            f"({eigs[0]:.3e}) within numerical tolerance. Correcting.",
            RuntimeWarning,
        )

        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0.0)
        cov = vecs @ np.diag(vals) @ vecs.T

    return cov


def _phi_variance_from_samples(
    U: float,
    z: np.ndarray,
    uu: float,
    vv: float,
    ww: float,
) -> float:
    """
    Evaluate the unbiased sample variance of phi using a common
    standard-normal sample z.

    The covariance is diagonal in the anisotropy family.
    """

    u_p = np.sqrt(uu) * z[:, 0]
    v_p = np.sqrt(vv) * z[:, 1]
    w_p = np.sqrt(ww) * z[:, 2]

    speed = np.sqrt(
        (U + u_p)**2 + v_p**2 + w_p**2
    )

    phi = np.arcsinh(speed)

    # Subtracting a constant does not change the variance.
    y = phi - math.asinh(abs(U))

    return float(np.var(y, ddof=1))


def mc_surrogate(
    U,
    uu,
    vv,
    ww,
    uv=0.0,
    n=MC_SAMPLES,
    seed=MC_SEED,
) -> dict:

    if n < 2:
        raise ValueError("n must be >= 2.")

    if min(uu, vv, ww) < 0.0:
        raise ValueError(
            "Variance inputs must be non-negative."
        )

    cov = _build_covariance(uu, vv, ww, uv)
    eigs = np.linalg.eigvalsh(cov)

    rng = np.random.default_rng(seed)

    samples = rng.multivariate_normal(
        mean=np.zeros(3),
        cov=cov,
        size=n,
        check_valid="raise",
    )

    u_p = samples[:, 0]
    v_p = samples[:, 1]
    w_p = samples[:, 2]

    speed = np.sqrt(
        (U + u_p)**2 + v_p**2 + w_p**2
    )

    phi = np.arcsinh(speed)

    shift = math.asinh(abs(U))
    y = phi - shift

    m1 = float(np.mean(y))
    m2 = float(np.mean(y**2))
    m3 = float(np.mean(y**3))
    m4 = float(np.mean(y**4))

    c2 = max(m2 - m1**2, 0.0)

    c4 = (
        m4 - 4.0 * m1 * m3
        + 6.0 * m1**2 * m2
        - 3.0 * m1**4
    )

    c4 = max(c4, 0.0)

    s2 = c2 * n / (n - 1)

    factor = (n - 3.0) / (n - 1.0)

    var_s2 = max(
        (c4 - factor * s2 * s2) / n,
        0.0,
    )

    se_s2 = math.sqrt(var_s2)

    phi_mean = shift + m1
    phi_prime = phi - phi_mean

    cosh_samples = np.cosh(2.0 * phi_prime)
    sinh_samples = np.sinh(2.0 * phi_prime)

    return {
        "phi_mean": float(phi_mean),
        "phi_sq": float(s2),
        "phi_sq_se": float(se_s2),
        "cosh": float(np.mean(cosh_samples)),
        "cosh_se": float(
            np.std(cosh_samples, ddof=1) / math.sqrt(n)
        ),
        "sinh": float(np.mean(sinh_samples)),
        "sinh_se": float(
            np.std(sinh_samples, ddof=1) / math.sqrt(n)
        ),
        "covariance_eigenvalues": [
            float(v) for v in eigs
        ],
    }


# =============================================================================
# §5 DNS PROFILE ANALYSIS
# =============================================================================

def analyse_dns(Re_tau: int) -> list[dict]:
    dns = load_dns(Re_tau)
    rows = []

    for i in range(dns["y_plus"].size):

        yp = float(dns["y_plus"][i])

        if not (Y_PLUS_MIN < yp < Y_PLUS_MAX):
            continue

        U = float(dns["U_plus"][i])
        uu = float(dns["uu"][i])
        vv = float(dns["vv"][i])
        ww = float(dns["ww"][i])
        uv = float(dns["uv"][i])

        k = 0.5 * (uu + vv + ww)

        if k < 0.0:
            raise ValueError(
                f"Negative k at Re={Re_tau}, y+={yp}"
            )

        mc = mc_surrogate(
            U, uu, vv, ww, uv,
            n=MC_SAMPLES,
            seed=MC_SEED,
        )

        delta_phi = mc["phi_mean"] - math.asinh(U)

        sinh_ratio = (
            mc["sinh"] / mc["cosh"]
            if mc["cosh"] != 0.0
            else float("nan")
        )

        cosh_ratio = (
            mc["cosh"] / math.exp(2.0 * mc["phi_sq"])
            if mc["phi_sq"] > 0.0
            else float("nan")
        )

        cosh_2phi_mean = math.cosh(
            2.0 * mc["phi_mean"]
        )

        sinh_2phi_mean = math.sinh(
            2.0 * mc["phi_mean"]
        )

        cosh_no_A2 = (
            rhs_identity(U, k)
            / max(cosh_2phi_mean, 1e-300)
        )

        phi_sq_A1 = 0.5 * math.log(
            max(cosh_no_A2, 1e-300)
        )

        cosh_after_A2 = (
            rhs_identity(U, k)
            - sinh_2phi_mean * mc["sinh"]
        ) / max(cosh_2phi_mean, 1e-300)

        phi_sq_A1A2 = 0.5 * math.log(
            max(cosh_after_A2, 1e-300)
        )

        rows.append({
            "Re": Re_tau,
            "y_plus": yp,
            "U": U,
            "k": k,
            "uu": uu,
            "vv": vv,
            "ww": ww,
            "uv": uv,
            "phi_sq_mc": mc["phi_sq"],
            "phi_sq_mc_se": mc["phi_sq_se"],
            "cosh_mc": mc["cosh"],
            "sinh_mc": mc["sinh"],
            "F1": F1_linear_u(U, uu),
            "F2": F2_isotropic(U, k),
            "F3": F3_small_fluct(U, k),
            "F4": F4_closed(U, k),
            "phi_sq_A1": phi_sq_A1,
            "phi_sq_A1A2": phi_sq_A1A2,
            "delta_phi": delta_phi,
            "sinh_ratio": sinh_ratio,
            "cosh_ratio": cosh_ratio,
        })

    return rows


# =============================================================================
# §6 ANISOTROPY SWEEP WITH COMMON RANDOM NUMBERS
# =============================================================================

def _anisotropy_sweep(
    U: float,
    k: float,
    A_vals: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    """
    Anisotropy sweep using common random numbers.

    The same standard-normal realization is used at every A value.
    This induces a deliberate coupling across the grid and enables
    paired Monte Carlo uncertainty estimates for adjacent differences.
    """

    rng = np.random.default_rng(seed)

    # Common random numbers across all anisotropy values.
    z = rng.standard_normal((n, 3))

    total = 2.0 * k
    vals = np.empty(len(A_vals), dtype=float)

    for j, A in enumerate(A_vals):

        uu = total / (1.0 + A)
        vv = ww = (total - uu) / 2.0

        vals[j] = _phi_variance_from_samples(
            U, z, uu, vv, ww
        )

    return vals


def _anisotropy_sweep_multiseed(
    U: float,
    k: float,
    A_vals: np.ndarray,
    n: int,
    seeds: tuple,
) -> dict:

    all_curves = np.zeros(
        (len(seeds), len(A_vals)),
        dtype=float,
    )

    for i, seed in enumerate(seeds):
        all_curves[i, :] = _anisotropy_sweep(
            U, k, A_vals, n, seed
        )

    return {
        "mean": all_curves.mean(axis=0),
        "sd": all_curves.std(axis=0, ddof=1),
        "all": all_curves,
    }


def _range_stats(vals: np.ndarray) -> dict:

    vmin = float(vals.min())
    vmax = float(vals.max())
    spread = vmax - vmin

    return {
        "min": vmin,
        "max": vmax,
        "spread": spread,
        "bound": 0.5 * spread,
    }


def _check_monotone_decreasing(
    values: np.ndarray,
    tolerance: float = 1e-8,
) -> dict:

    values = np.asarray(values, dtype=float)
    differences = np.diff(values)

    violations = np.where(
        differences > tolerance
    )[0]

    return {
        "monotone_nonincreasing": bool(
            violations.size == 0
        ),
        "max_upward_step": float(
            max(0.0, differences.max())
        ) if differences.size else 0.0,
        "n_violations": int(violations.size),
        "tolerance": float(tolerance),
    }


# =============================================================================
# §6A PAIRED LOCAL-MONOTONICITY DIAGNOSTIC
# =============================================================================

def paired_monotonicity_diagnostic(
    U: float,
    k: float,
    A_vals: np.ndarray,
    seed: int,
    n: int = PAIRED_DIAGNOSTIC_SAMPLES,
    alpha: float = PAIRED_ALPHA,
) -> dict:
    """
    Paired diagnostic for adjacent upward steps in g(A).

    For each adjacent pair:
        Delta_i = g(A_{i+1}) - g(A_i)

    The influence function for the population variance is:
        IF = (Y - E[Y])^2 - Var(Y)

    The paired influence-function difference is:
        IF_delta = IF_{i+1} - IF_i

    Its sample standard deviation estimates the paired standard
    error of the adjacent variance difference.

    The Bonferroni threshold controls the family-wise error rate
    across all adjacent intervals in this one curve.
    """

    if n < 2:
        raise ValueError("n must be >= 2.")

    A_vals = np.asarray(A_vals, dtype=float)

    if A_vals.ndim != 1 or len(A_vals) < 2:
        raise ValueError(
            "A_vals must be a one-dimensional grid with >=2 points."
        )

    rng = np.random.default_rng(seed)

    # The same standard-normal triplets are used at every A.
    z = rng.standard_normal((n, 3))

    total = 2.0 * k

    y_values = []
    g_values = []
    influence_values = []

    for A in A_vals:

        uu = total / (1.0 + A)
        vv = ww = (total - uu) / 2.0

        u_p = np.sqrt(uu) * z[:, 0]
        v_p = np.sqrt(vv) * z[:, 1]
        w_p = np.sqrt(ww) * z[:, 2]

        speed = np.sqrt(
            (U + u_p)**2 + v_p**2 + w_p**2
        )

        phi = np.arcsinh(speed)

        # A constant shift does not affect variance.
        y = phi - math.asinh(abs(U))

        y_bar = float(np.mean(y))
        centered = y - y_bar

        # Unbiased sample variance, ddof=1.
        g = float(np.var(y, ddof=1))

        # Plug-in influence function for the variance.
        # Centering the squared deviations gives mean(IF) = 0.
        squared = centered**2
        influence = squared - float(np.mean(squared))

        y_values.append(y)
        g_values.append(g)
        influence_values.append(influence)

    g_values = np.asarray(g_values)
    influence_values = np.asarray(influence_values)

    m = len(A_vals) - 1

    # Pointwise two-sided normal critical value.
    z_pointwise = float(
        norm.ppf(1.0 - alpha / 2.0)
    )

    # Bonferroni family-wise critical value for m intervals.
    z_bonferroni = float(
        norm.ppf(1.0 - alpha / (2.0 * m))
    )

    intervals = []

    for i in range(m):

        delta = float(
            g_values[i + 1] - g_values[i]
        )

        # Paired influence-function difference.
        influence_delta = (
            influence_values[i + 1]
            - influence_values[i]
        )

        paired_se = float(
            np.std(influence_delta, ddof=1)
            / math.sqrt(n)
        )

        if paired_se > 0.0:
            z_score = delta / paired_se
        else:
            z_score = (
                math.copysign(float("inf"), delta)
                if delta != 0.0
                else 0.0
            )

        pointwise_lo = (
            delta - z_pointwise * paired_se
        )

        pointwise_hi = (
            delta + z_pointwise * paired_se
        )

        bonf_lo = (
            delta - z_bonferroni * paired_se
        )

        bonf_hi = (
            delta + z_bonferroni * paired_se
        )

        intervals.append({
            "index": int(i),
            "A_left": float(A_vals[i]),
            "A_right": float(A_vals[i + 1]),
            "delta_g": delta,
            "paired_se": paired_se,
            "z_score": float(z_score),

            "pointwise_ci_95": [
                float(pointwise_lo),
                float(pointwise_hi),
            ],

            "bonferroni_ci_95": [
                float(bonf_lo),
                float(bonf_hi),
            ],

            "pointwise_significant_upward": bool(
                delta > 0.0
                and pointwise_lo > 0.0
            ),

            "bonferroni_significant_upward": bool(
                delta > 0.0
                and bonf_lo > 0.0
            ),
        })

    upward = [
        item for item in intervals
        if item["delta_g"] > 0.0
    ]

    largest_upward = (
        max(upward, key=lambda x: x["delta_g"])
        if upward else None
    )

    largest_standardized = (
        max(
            intervals,
            key=lambda x: x["z_score"]
        )
        if intervals else None
    )

    return {
        "seed": int(seed),
        "U": float(U),
        "k": float(k),
        "n": int(n),
        "n_intervals": int(m),
        "alpha": float(alpha),
        "critical_z_pointwise": z_pointwise,
        "critical_z_bonferroni": z_bonferroni,
        "g_values": [
            float(v) for v in g_values
        ],
        "largest_upward_step": largest_upward,
        "largest_standardized_step": largest_standardized,
        "n_upward_steps": int(len(upward)),
        "n_pointwise_significant_upward": int(
            sum(
                x["pointwise_significant_upward"]
                for x in intervals
            )
        ),
        "n_bonferroni_significant_upward": int(
            sum(
                x["bonferroni_significant_upward"]
                for x in intervals
            )
        ),
        "intervals": intervals,
        "interpretation": (
            "Paired Monte Carlo diagnostic under the specified "
            "Gaussian anisotropy family. Pointwise and Bonferroni "
            "intervals quantify Monte Carlo sampling uncertainty; "
            "they do not establish physical non-monotonicity."
        ),
    }


def print_paired_monotonicity_diagnostic(
    result: dict,
) -> None:

    print(
        f"\n  Paired local-monotonicity diagnostic "
        f"(seed={result['seed']}):"
    )

    print(
        f"    N={result['n']:,}, "
        f"adjacent intervals={result['n_intervals']}"
    )

    print(
        f"    Critical z: pointwise="
        f"{result['critical_z_pointwise']:.4f}, "
        f"Bonferroni="
        f"{result['critical_z_bonferroni']:.4f}"
    )

    largest = result["largest_upward_step"]

    if largest is None:
        print("    No upward step was found.")
    else:
        print(
            f"    Largest upward step:"
            f" A={largest['A_left']:.4f} -> "
            f"{largest['A_right']:.4f}"
        )

        print(
            f"      Delta g = {largest['delta_g']:.8e}"
        )

        print(
            f"      Paired SE = {largest['paired_se']:.8e}"
        )

        print(
            f"      z-score = {largest['z_score']:.4f}"
        )

        print(
            f"      Pointwise 95% CI = "
            f"[{largest['pointwise_ci_95'][0]:.8e}, "
            f"{largest['pointwise_ci_95'][1]:.8e}]"
        )

        print(
            f"      Bonferroni 95% CI = "
            f"[{largest['bonferroni_ci_95'][0]:.8e}, "
            f"{largest['bonferroni_ci_95'][1]:.8e}]"
        )

        print(
            f"      Pointwise significant upward step: "
            f"{largest['pointwise_significant_upward']}"
        )

        print(
            f"      Bonferroni significant upward step: "
            f"{largest['bonferroni_significant_upward']}"
        )

    print(
        f"    Total upward steps: "
        f"{result['n_upward_steps']}"
    )

    print(
        f"    Pointwise significant upward steps: "
        f"{result['n_pointwise_significant_upward']}"
    )

    print(
        f"    Bonferroni significant upward steps: "
        f"{result['n_bonferroni_significant_upward']}"
    )


# =============================================================================
# §7 MULTI-SEED INFORMATION BARRIER
# =============================================================================

def barrier_multiseed(
    U: float = 8.0,
    k: float = 4.0,
    seeds: tuple = ROBUST_SEEDS,
    n: int = ROBUST_SAMPLES,
) -> dict:

    A_vals_main = np.linspace(
        A_MIN, A_MAX, A_POINTS
    )

    A_vals_ext = np.linspace(
        A_MIN, A_MAX_EXTENDED, A_POINTS_EXTENDED
    )

    per_seed = []

    main_spreads = []
    main_bounds = []
    main_mins = []
    main_maxs = []

    ext_spreads = []
    ext_bounds = []
    ext_mins = []
    ext_maxs = []

    main_curves = []
    ext_curves = []
    monotonicity = []

    for seed in seeds:

        vals_main = _anisotropy_sweep(
            U, k, A_vals_main, n, seed
        )

        vals_ext = _anisotropy_sweep(
            U, k, A_vals_ext, n, seed
        )

        main_curves.append(vals_main)
        ext_curves.append(vals_ext)

        rs_main = _range_stats(vals_main)
        rs_ext = _range_stats(vals_ext)

        mono_main = _check_monotone_decreasing(
            vals_main
        )

        mono_ext = _check_monotone_decreasing(
            vals_ext
        )

        monotonicity.append({
            "seed": int(seed),
            "main": mono_main,
            "extended": mono_ext,
        })

        per_seed.append({
            "seed": int(seed),
            "main": rs_main,
            "extended": rs_ext,
            "monotonicity_main": mono_main,
            "monotonicity_extended": mono_ext,
        })

        main_mins.append(rs_main["min"])
        main_maxs.append(rs_main["max"])
        main_spreads.append(rs_main["spread"])
        main_bounds.append(rs_main["bound"])

        ext_mins.append(rs_ext["min"])
        ext_maxs.append(rs_ext["max"])
        ext_spreads.append(rs_ext["spread"])
        ext_bounds.append(rs_ext["bound"])

    def _agg(x):
        m, s, lo, hi = _stats(x)
        return {
            "mean": m,
            "sd": s,
            "ci": [lo, hi],
        }

    return {
        "U": float(U),
        "k": float(k),
        "A_main": [A_MIN, A_MAX],
        "A_extended": [A_MIN, A_MAX_EXTENDED],
        "seeds": list(seeds),
        "n_per_run": int(n),
        "per_seed": per_seed,
        "monotonicity": monotonicity,

        "monotonicity_mean_main": (
            _check_monotone_decreasing(
                np.mean(main_curves, axis=0)
            )
        ),

        "monotonicity_mean_extended": (
            _check_monotone_decreasing(
                np.mean(ext_curves, axis=0)
            )
        ),

        "main": {
            "min_mean": float(np.mean(main_mins)),
            "max_mean": float(np.mean(main_maxs)),
            "spread": _agg(main_spreads),
            "bound": _agg(main_bounds),
        },

        "extended": {
            "min_mean": float(np.mean(ext_mins)),
            "max_mean": float(np.mean(ext_maxs)),
            "spread": _agg(ext_spreads),
            "bound": _agg(ext_bounds),
        },
    }


# =============================================================================
# §8 CONVERGENCE STUDY
# =============================================================================

def _t_critical(df: int) -> float:

    table = {
        1: 12.706, 2: 4.303, 3: 3.182,
        4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086,
    }

    return table.get(
        df,
        1.96 if df > 20 else 2.0
    )


def _stats(values):

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return (
            float(np.mean(x)) if len(x) else float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    se = s / math.sqrt(len(x))
    t = _t_critical(len(x) - 1)

    return (
        m,
        s,
        m - t * se,
        m + t * se,
    )


def convergence_study(
    U: float = 8.0,
    k: float = 4.0,
) -> dict:

    print(
        f"\n--- Convergence study: U={U}, k={k} ---"
    )

    results = []

    for A in CONV_A_VALUES:

        uu = 2.0 * k / (1.0 + A)
        vv = ww = (2.0 * k - uu) / 2.0

        for N in CONV_SAMPLE_SIZES:

            per_seed = [
                mc_surrogate(
                    U, uu, vv, ww,
                    n=N, seed=s
                )["phi_sq"]
                for s in CONV_SEEDS
            ]

            m, s, lo, hi = _stats(per_seed)

            results.append({
                "A": float(A),
                "N": int(N),
                "mean": m,
                "seed_sd": s,
                "ci_lo": lo,
                "ci_hi": hi,
            })

            print(
                f"  A={A:.2f}  N={N:>9,}  "
                f"mean={m:.6f}  SD={s:.2e}"
            )

    return {"results": results}


# =============================================================================
# FIGURES
# =============================================================================

def fig_phi_variance(rows, path):

    fig, axes = plt.subplots(
        1, len(RE_LIST),
        figsize=(5.5 * len(RE_LIST), 4.2)
    )

    if len(RE_LIST) == 1:
        axes = [axes]

    for ax, Re in zip(axes, RE_LIST):

        sub = [
            r for r in rows
            if r["Re"] == Re
        ]

        if not sub:
            continue

        yp = [r["y_plus"] for r in sub]

        ax.loglog(
            yp,
            [r["phi_sq_mc"] for r in sub],
            "k-",
            label="Gaussian surrogate",
            lw=2,
        )

        ax.loglog(
            yp, [r["F1"] for r in sub],
            "b--", label=r"$F_1$"
        )

        ax.loglog(
            yp, [r["F2"] for r in sub],
            "g:", label=r"$F_2$"
        )

        ax.loglog(
            yp, [r["F3"] for r in sub],
            "m--", label=r"$F_3$"
        )

        ax.loglog(
            yp, [r["F4"] for r in sub],
            "r-.", label=r"$F_4$"
        )

        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\langle\phi'^2\rangle$")
        ax.set_title(rf"$Re_\tau = {Re}$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_assumptions(rows, path):

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.2)
    )

    for Re in RE_LIST:

        sub = [
            r for r in rows
            if r["Re"] == Re
        ]

        if not sub:
            continue

        yp = [r["y_plus"] for r in sub]

        axes[0].plot(
            yp,
            [r["delta_phi"] for r in sub],
            "o-", ms=3, label=rf"$Re={Re}$"
        )

        axes[1].plot(
            yp,
            [r["sinh_ratio"] for r in sub],
            "o-", ms=3, label=rf"$Re={Re}$"
        )

        axes[2].plot(
            yp,
            [r["cosh_ratio"] for r in sub],
            "o-", ms=3, label=rf"$Re={Re}$"
        )

    labels = [
        "A1: mean shift",
        "A2: hyperbolic asymmetry",
        "A3: Gaussian-moment consistency",
    ]

    ylabels = [
        r"$\langle\phi\rangle-\operatorname{arcsinh}U$",
        r"$\langle\sinh 2\phi'\rangle/"
        r"\langle\cosh 2\phi'\rangle$",
        r"$\langle\cosh 2\phi'\rangle/"
        r"e^{2\langle\phi'^2\rangle}$",
    ]

    for ax, title, ylabel in zip(
        axes, labels, ylabels
    ):

        ref = (
            0.0
            if ("mean" in title or "asymmetry" in title)
            else 1.0
        )

        ax.axhline(
            ref, color="k", ls="--", alpha=0.4
        )

        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_barrier(
    res: dict,
    path: str,
    U: float = 8.0,
    k: float = 4.0,
    seeds: tuple = ROBUST_SEEDS,
) -> None:

    A_vals = np.linspace(
        A_MIN, A_MAX, A_POINTS
    )

    sweep = _anisotropy_sweep_multiseed(
        U, k, A_vals, ROBUST_SAMPLES, seeds
    )

    phi_mean = sweep["mean"]
    phi_sd = sweep["sd"]

    vmin = float(phi_mean.min())
    vmax = float(phi_mean.max())
    spread = vmax - vmin
    bound = 0.5 * spread

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        A_vals,
        phi_mean,
        "o-",
        color="C0",
        label=(
            r"$\langle\phi'^2\rangle$ (mean of "
            f"{len(seeds)} seeds)"
        ),
    )

    ax.fill_between(
        A_vals,
        phi_mean - phi_sd,
        phi_mean + phi_sd,
        color="C0",
        alpha=0.20,
        label=r"$\pm 1\sigma$ across seeds",
    )

    ax.axhline(
        vmin, color="gray", ls=":", alpha=0.6
    )

    ax.axhline(
        vmax, color="gray", ls=":", alpha=0.6
    )

    ax.fill_between(
        [A_vals.min(), A_vals.max()],
        vmin, vmax,
        color="C1",
        alpha=0.10,
        label=rf"spread = {spread:.4f}",
    )

    ax.axhline(
        vmin + bound,
        color="C3",
        ls="--",
        label=rf"minimax bound = {bound:.4f}",
    )

    ax.set_xlabel(
        r"Anisotropy ratio $A = "
        r"(\langle v'^2\rangle+\langle w'^2\rangle)"
        r"/\langle u'^2\rangle$"
    )

    ax.set_ylabel(r"$\langle\phi'^2\rangle$")

    ax.set_title(
        rf"Information barrier at $U={U}$, $k={k}$ "
        rf"(mean of {len(seeds)} seeds)"
    )

    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_barrier_extended(
    res: dict,
    path: str,
    U: float = 8.0,
    k: float = 4.0,
    seeds: tuple = ROBUST_SEEDS,
) -> dict:

    A_vals = np.linspace(
        A_MIN, A_MAX_EXTENDED,
        A_POINTS_EXTENDED
    )

    sweep = _anisotropy_sweep_multiseed(
        U, k, A_vals, ROBUST_SAMPLES, seeds
    )

    phi_mean = sweep["mean"]
    phi_sd = sweep["sd"]

    monotonicity_mean = _check_monotone_decreasing(
        phi_mean
    )

    mask_main = A_vals <= A_MAX + 1e-10

    vmin_main = float(
        phi_mean[mask_main].min()
    )

    vmax_main = float(
        phi_mean[mask_main].max()
    )

    spread_main = vmax_main - vmin_main

    vmin_ext = float(phi_mean.min())
    vmax_ext = float(phi_mean.max())
    spread_ext = vmax_ext - vmin_ext

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.plot(
        A_vals,
        phi_mean,
        "k-",
        lw=1.8,
        label=(
            r"$\langle\phi'^2\rangle$ "
            f"(mean of {len(seeds)} seeds)"
        ),
    )

    ax.fill_between(
        A_vals,
        phi_mean - phi_sd,
        phi_mean + phi_sd,
        color="gray",
        alpha=0.20,
        label=r"$\pm 1\sigma$ across seeds",
    )

    ax.axvspan(
        0.0, A_MAX,
        alpha=0.12,
        color="C0",
        label=rf"Main study range $[0,{A_MAX:g}]$",
    )

    ax.axhline(
        vmin_main,
        color="C0",
        ls=":",
        lw=1.2,
        label=rf"$\min$ on $[0,{A_MAX:g}]$: {vmin_main:.4f}",
    )

    ax.axhline(
        vmin_ext,
        color="C3",
        ls=":",
        lw=1.2,
        label=(
            rf"$\min$ on $[0,{A_MAX_EXTENDED:g}]$: "
            f"{vmin_ext:.4f}"
        ),
    )

    ax.annotate(
        "",
        xy=(A_MAX + 0.2, vmax_main),
        xytext=(A_MAX + 0.2, vmin_main),
        arrowprops=dict(
            arrowstyle="<->",
            color="C0",
            lw=1.4,
        ),
    )

    ax.text(
        A_MAX + 0.35,
        0.5 * (vmax_main + vmin_main),
        rf"$\Delta_{{[0,{A_MAX:g}]}} = {spread_main:.3f}$",
        color="C0",
        fontsize=10,
        va="center",
    )

    ax.annotate(
        "",
        xy=(A_MAX_EXTENDED - 0.5, vmax_ext),
        xytext=(A_MAX_EXTENDED - 0.5, vmin_ext),
        arrowprops=dict(
            arrowstyle="<->",
            color="C3",
            lw=1.4,
        ),
    )

    ax.text(
        A_MAX_EXTENDED - 1.8,
        0.5 * (vmax_ext + vmin_ext),
        (
            rf"$\Delta_{{[0,{A_MAX_EXTENDED:g}]}}"
            rf" = {spread_ext:.3f}$"
        ),
        color="C3",
        fontsize=10,
        va="center",
        ha="right",
    )

    ax.set_xlabel(
        r"Anisotropy ratio $A = "
        r"(\langle v'^2\rangle+\langle w'^2\rangle)"
        r"/\langle u'^2\rangle$"
    )

    ax.set_ylabel(r"$\langle\phi'^2\rangle$")

    ax.set_title(
        rf"Extended anisotropy sweep at $U={U}$, $k={k}$ "
        rf"(mean of {len(seeds)} seeds)"
    )

    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return {
        "A_max_main": A_MAX,
        "A_max_extended": A_MAX_EXTENDED,
        "vmin_main": vmin_main,
        "vmax_main": vmax_main,
        "spread_main": spread_main,
        "vmin_ext": vmin_ext,
        "vmax_ext": vmax_ext,
        "spread_ext": spread_ext,
        "growth_factor": spread_ext / spread_main,
        "seeds_used": list(seeds),
        "monotonicity_mean": monotonicity_mean,
    }


def fig_convergence(conv: dict, path: str) -> None:

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for A in CONV_A_VALUES:

        sub = [
            r for r in conv["results"]
            if r["A"] == A
        ]

        if not sub:
            continue

        ax.semilogx(
            [r["N"] for r in sub],
            [r["mean"] for r in sub],
            "o-",
            label=f"A={A:g}",
        )

    ax.set_xlabel(
        r"Monte Carlo sample size $N$"
    )

    ax.set_ylabel(
        r"Estimated $\langle\phi'^2\rangle$"
    )

    ax.set_title(
        "Nested sample-size convergence"
    )

    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# EXPORTS
# =============================================================================

def export_csv(rows, path):

    if not rows:
        return

    with open(
        path, "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        w.writeheader()
        w.writerows(rows)


def export_barrier_multiseed(res, path):

    keys = [
        "seed",
        "main_min", "main_max",
        "main_spread", "main_bound",
        "ext_min", "ext_max",
        "ext_spread", "ext_bound",
    ]

    with open(
        path, "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        w.writeheader()

        for p in res["per_seed"]:

            w.writerow({
                "seed": p["seed"],
                "main_min": p["main"]["min"],
                "main_max": p["main"]["max"],
                "main_spread": p["main"]["spread"],
                "main_bound": p["main"]["bound"],
                "ext_min": p["extended"]["min"],
                "ext_max": p["extended"]["max"],
                "ext_spread": p["extended"]["spread"],
                "ext_bound": p["extended"]["bound"],
            })


def export_paired_diagnostics(
    diagnostics: dict,
    path: str,
) -> None:
    """
    Export all adjacent-interval paired diagnostics to CSV.
    """

    rows = []

    for seed_key, result in diagnostics.items():

        for item in result["intervals"]:

            rows.append({
                "seed": result["seed"],
                "A_left": item["A_left"],
                "A_right": item["A_right"],
                "delta_g": item["delta_g"],
                "paired_se": item["paired_se"],
                "z_score": item["z_score"],
                "pointwise_ci_lo": item["pointwise_ci_95"][0],
                "pointwise_ci_hi": item["pointwise_ci_95"][1],
                "bonferroni_ci_lo": item["bonferroni_ci_95"][0],
                "bonferroni_ci_hi": item["bonferroni_ci_95"][1],
                "pointwise_significant_upward":
                    item["pointwise_significant_upward"],
                "bonferroni_significant_upward":
                    item["bonferroni_significant_upward"],
            })

    if not rows:
        return

    export_csv(rows, path)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        " Information Barrier of Algebraic Wall Functions — Supplementary"
    )
    print("=" * 78)

    prepare_data()

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # -------------------------------------------------------------------------
    # §5 DNS comparison
    # -------------------------------------------------------------------------

    print(
        "\n--- DNS comparison (F1-F4 vs surrogate) ---"
    )

    all_rows = []

    for Re in RE_LIST:

        print(f"  Re_tau = {Re}")

        try:
            rows = analyse_dns(Re)

        except FileNotFoundError as exc:
            print(f"    Skipped: {exc}")
            continue

        all_rows.extend(rows)

        print(f"    {len(rows)} stations.")

    if all_rows:

        export_csv(
            all_rows,
            "results/dns_comparison.csv",
        )

        fig_phi_variance(
            all_rows,
            "figures/fig1_phi_variance.pdf",
        )

        fig_assumptions(
            all_rows,
            "figures/fig2_assumptions.pdf",
        )

    # -------------------------------------------------------------------------
    # §7 Multi-seed barrier
    # -------------------------------------------------------------------------

    print(
        "\n--- Multi-seed information-barrier experiment ---"
    )

    res = barrier_multiseed(
        U=8.0,
        k=4.0,
    )

    print(
        f"  U={res['U']}  k={res['k']}"
    )

    print("  Per-seed:")

    for p in res["per_seed"]:

        print(
            f"    seed={p['seed']:>4}  "
            f"[0,2]: spread={p['main']['spread']:.6f}  "
            f"bound={p['main']['bound']:.6f}  |  "
            f"[0,10]: spread={p['extended']['spread']:.6f}  "
            f"bound={p['extended']['bound']:.6f}"
        )

    m = res["main"]
    e = res["extended"]

    print(f"\n  MAIN [0, {A_MAX:g}]:")

    print(
        f"    min={m['min_mean']:.4f}  "
        f"max={m['max_mean']:.4f}"
    )

    print(
        f"    spread = {m['spread']['mean']:.6f} "
        f"(95% t-CI: "
        f"[{m['spread']['ci'][0]:.6f}, "
        f"{m['spread']['ci'][1]:.6f}])"
    )

    print(
        f"    bound  = {m['bound']['mean']:.6f} "
        f"(95% t-CI: "
        f"[{m['bound']['ci'][0]:.6f}, "
        f"{m['bound']['ci'][1]:.6f}])"
    )

    print(
        f"\n  EXTENDED [0, {A_MAX_EXTENDED:g}]:"
    )

    print(
        f"    min={e['min_mean']:.4f}  "
        f"max={e['max_mean']:.4f}"
    )

    print(
        f"    spread = {e['spread']['mean']:.6f} "
        f"(95% t-CI: "
        f"[{e['spread']['ci'][0]:.6f}, "
        f"{e['spread']['ci'][1]:.6f}])"
    )

    print(
        f"    bound  = {e['bound']['mean']:.6f} "
        f"(95% t-CI: "
        f"[{e['bound']['ci'][0]:.6f}, "
        f"{e['bound']['ci'][1]:.6f}])"
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    fig_barrier(
        res,
        "figures/fig3_barrier.pdf",
        U=8.0,
        k=4.0,
    )

    ext_info = fig_barrier_extended(
        res,
        "figures/fig5_barrier_extended.pdf",
        U=8.0,
        k=4.0,
    )

    print(
        f"\n  Figure 5 (extended sweep, mean of "
        f"{len(ROBUST_SEEDS)} seeds):"
    )

    print(
        f"    Growth factor = "
        f"{ext_info['growth_factor']:.3f}x"
    )

    print(
        "    Sampled monotonicity checks (non-increasing):"
    )

    for item in res["monotonicity"]:

        print(
            f"      seed={item['seed']}: "
            f"[0,2]={item['main']['monotone_nonincreasing']} "
            f"[0,10]={item['extended']['monotone_nonincreasing']}"
        )

    print(
        f"      seed-mean: "
        f"[0,2]={res['monotonicity_mean_main']['monotone_nonincreasing']} "
        f"[0,10]={res['monotonicity_mean_extended']['monotone_nonincreasing']}"
    )

    print(
        f"    [0,{A_MAX:g}] sampled spread = "
        f"{ext_info['spread_main']:.4f} "
        f"(multi-seed: {m['spread']['mean']:.4f})"
    )

    print(
        f"    [0,{A_MAX_EXTENDED:g}] sampled spread = "
        f"{ext_info['spread_ext']:.4f} "
        f"(multi-seed: {e['spread']['mean']:.4f})"
    )

    # -------------------------------------------------------------------------
    # Consistency check
    # -------------------------------------------------------------------------

    delta_main = abs(
        ext_info["spread_main"]
        - m["spread"]["mean"]
    )

    delta_ext = abs(
        ext_info["spread_ext"]
        - e["spread"]["mean"]
    )

    tol = 1e-3

    consistent = (
        delta_main < tol
        and delta_ext < tol
    )

    print(
        f"    Consistency check: "
        f"|Delta_fig5 - Delta_multiseed| = "
        f"{delta_main:.2e} (main), "
        f"{delta_ext:.2e} (extended)"
    )

    if consistent:
        print(
            "    --> OK: the two sampled-range summaries "
            "agree within tolerance."
        )
    else:
        print(
            f"    --> WARNING: sampled-summary discrepancy "
            f"(tol={tol})."
        )

    print(
        "    Note: this is a consistency check between two "
        "summaries of the same Monte Carlo curves, not an "
        "independent validation."
    )

    print(
        "    The 95% t-intervals quantify seed variability only, "
        "not model-form uncertainty."
    )

    export_barrier_multiseed(
        res,
        "results/barrier_multiseed.csv",
    )

    # -------------------------------------------------------------------------
    # §6A Paired local-monotonicity diagnostic
    # -------------------------------------------------------------------------

    print(
        "\n--- Paired local-monotonicity diagnostic ---"
    )

    A_vals_diagnostic = np.linspace(
        A_MIN,
        A_MAX_EXTENDED,
        A_POINTS_EXTENDED,
    )

    paired_diagnostics = {}

    for seed in PAIRED_DIAGNOSTIC_SEEDS:

        diagnostic = paired_monotonicity_diagnostic(
            U=8.0,
            k=4.0,
            A_vals=A_vals_diagnostic,
            seed=seed,
            n=PAIRED_DIAGNOSTIC_SAMPLES,
            alpha=PAIRED_ALPHA,
        )

        paired_diagnostics[str(seed)] = diagnostic

        print_paired_monotonicity_diagnostic(
            diagnostic
        )

    export_paired_diagnostics(
        paired_diagnostics,
        "results/paired_monotonicity_diagnostics.csv",
    )

    # -------------------------------------------------------------------------
    # §8 Convergence
    # -------------------------------------------------------------------------

    conv = convergence_study(
        U=8.0,
        k=4.0,
    )

    export_csv(
        conv["results"],
        "results/convergence.csv",
    )

    fig_convergence(
        conv,
        "figures/fig4_convergence.pdf",
    )

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------

    summary = {

        "RE_LIST": list(RE_LIST),

        "method": (
            "DNS-parameterized Gaussian Monte Carlo surrogate; "
            "not direct instantaneous DNS sampling"
        ),

        "anisotropy_sweep_sampling": {
            "method": "Common Random Numbers",
            "description": (
                "The same standard-normal realization is used "
                "at every anisotropy value within each seed."
            ),
        },

        "barrier": {
            "U": res["U"],
            "k": res["k"],
            "A_main": res["A_main"],
            "A_extended": res["A_extended"],
            "seeds": res["seeds"],
            "n_per_run": res["n_per_run"],
            "per_seed": res["per_seed"],
            "main_range": res["main"],
            "extended_range": res["extended"],
            "figure5_summary": ext_info,

            "monotonicity": {
                "per_seed": res["monotonicity"],
                "mean_main": res["monotonicity_mean_main"],
                "mean_extended": res["monotonicity_mean_extended"],
                "interpretation": (
                    "Sampled-grid check only; not a proof "
                    "of continuous monotonicity."
                ),
            },

            "confidence_interval_scope": (
                "Seed-to-seed variability only; excludes "
                "model-form uncertainty."
            ),
        },

        "paired_monotonicity_diagnostic": {
            "seeds": list(PAIRED_DIAGNOSTIC_SEEDS),
            "n": int(PAIRED_DIAGNOSTIC_SAMPLES),
            "alpha": float(PAIRED_ALPHA),
            "method": (
                "Paired influence-function standard error "
                "for adjacent differences in sample variance."
            ),
            "multiple_comparison_correction": (
                "Bonferroni family-wise correction over all "
                "adjacent intervals in each diagnostic curve."
            ),
            "results": paired_diagnostics,
            "interpretation": (
                "The diagnostic compares observed local upward "
                "steps with paired Monte Carlo sampling uncertainty. "
                "It does not establish physical non-monotonicity "
                "or quantify Gaussian model-form uncertainty."
            ),
        },

        "consistency_check": {
            "abs_delta_main": delta_main,
            "abs_delta_extended": delta_ext,
            "tolerance": tol,
            "passed": bool(consistent),
        },

        "n_dns_rows": len(all_rows),
    }

    with open(
        "results/summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=float,
        )

    # -------------------------------------------------------------------------
    # Output report
    # -------------------------------------------------------------------------

    print("\n" + "=" * 78)
    print(" Outputs written:")

    print("   figures/fig1_phi_variance.pdf")
    print("   figures/fig2_assumptions.pdf")
    print("   figures/fig3_barrier.pdf")
    print("   figures/fig4_convergence.pdf")
    print("   figures/fig5_barrier_extended.pdf")

    print("   results/dns_comparison.csv")
    print("   results/barrier_multiseed.csv")
    print("   results/convergence.csv")
    print("   results/paired_monotonicity_diagnostics.csv")
    print("   results/summary.json")

    print("=" * 78)


if __name__ == "__main__":
    main()
