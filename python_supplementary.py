#!/usr/bin/env python3
# =============================================================================
#  Supplementary Material for
#
#     "The information barrier of closed-form wall functions for
#      arcsinh-transformed velocity variance: why k and U cannot
#      determine <phi'^2>"
#
#  This script reproduces every numerical result, table, and figure
#  reported in the manuscript. It is organised into self-contained
#  sections, each corresponding to one part of the paper:
#
#     §0  Data preparation (local or Google Colab)   -> prepare_data()
#     §1  DNS data loading                            -> load_dns()
#     §2  The exact hyperbolic identity               -> rhs_identity()
#     §3  The naive closed form F4                    -> F4_closed()
#     §4  Numerical comparison against DNS            -> analyse_dns()
#     §5  Diagnosis of the three nonlinear mechanisms -> diagnose (in §4)
#     §6  The controlled information-barrier experiment -> barrier()
#
#  Author:      [corresponding author]
#  Affiliation: [institution]
#  Date:        2025
#
#  Requirements: numpy, scipy, matplotlib
#
#  Usage (local):
#     $ python supplementary.py
#
#  Usage (Google Colab):
#     Upload supplementary.py, run it. If the DNS folders are absent,
#     the script will prompt for a compressed archive (chandata.tar.gz
#     or chandata.zip) containing chan180/, chan395/, chan590/.
#
#  Input:
#     DNS channel-flow data of Moser, Kim & Mansour (1999) placed in
#     ./chan180/, ./chan395/, ./chan590/ (each containing a ./profiles/
#     subdirectory with the .reystress and .means files).
#
#  Output:
#     ./figures/fig1_phi_variance.pdf
#     ./figures/fig2_assumptions.pdf
#     ./figures/fig3_barrier.pdf
#     ./results/dns_comparison.csv
#     ./results/barrier.csv
#     ./results/summary.json
# =============================================================================

import os
import json
import tarfile
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
RE_LIST       = (180, 395, 590)   # friction Reynolds numbers of Moser et al.
MC_SAMPLES    = 200_000           # Monte Carlo sample size per point
RNG_SEED      = 42
Y_PLUS_MAX    = 60.0              # upper bound for near-wall analysis
Y_PLUS_MIN    = 0.5               # lower bound (exclude the wall singularity)
FONT          = {"family": "serif", "size": 10}

plt.rc("font", **FONT)
plt.rc("mathtext", fontset="cm")
plt.rc("axes", linewidth=0.8)
plt.rc("lines", linewidth=1.2)


# =============================================================================
# §0  Data preparation
#
# The dataset of Moser, Kim & Mansour (1999) must be available in three
# folders chan180/, chan395/, chan590/. This function handles three cases:
#   (i)  the folders are already present locally -> nothing to do;
#   (ii) the script runs in Google Colab and the folders are absent ->
#        prompt the user to upload a compressed archive and extract it;
#   (iii) the script runs locally and the folders are absent -> raise an
#        informative error with download instructions.
# =============================================================================
def prepare_data():
    """
    Ensure the DNS data folders are available.

    Returns
    -------
    None. Side effect: creates ./chan{180,395,590}/ in the working directory
    when running in Colab and the folders were absent.

    Raises
    ------
    FileNotFoundError
        When running locally and the folders are missing.
    RuntimeError
        When the uploaded archive does not contain the expected folders.
    """
    expected = [f"chan{Re}" for Re in RE_LIST]
    if all(os.path.isdir(f) for f in expected):
        print("[prepare_data] DNS data folders found locally.")
        return

    # Detect Google Colab
    in_colab = False
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except ImportError:
        pass

    if not in_colab:
        raise FileNotFoundError(
            "DNS data folders chan180/, chan395/, chan590/ are missing.\n"
            "Please download the channel-flow data of Moser, Kim & Mansour\n"
            "(1999) from the official source and place the three folders\n"
            "in the working directory, or run this script in Google Colab\n"
            "and upload a compressed archive (chandata.tar.gz or .zip)."
        )

    # --- Colab: prompt for upload --------------------------------------
    from google.colab import files
    print("[prepare_data] DNS data not found. "
          "Please upload chandata.tar.gz, chandata.tgz, or chandata.zip.")
    uploaded = files.upload()
    for fname in uploaded.keys():
        print(f"[prepare_data] Processing uploaded file: {fname}")
        if fname.endswith(".tar.gz") or fname.endswith(".tgz"):
            with tarfile.open(fname, "r:gz") as tar:
                tar.extractall(".")
            print(f"[prepare_data] Extracted TAR.GZ: {fname}")
            os.remove(fname)
        elif fname.endswith(".zip"):
            with zipfile.ZipFile(fname, "r") as zf:
                zf.extractall(".")
            print(f"[prepare_data] Extracted ZIP: {fname}")
            os.remove(fname)
        else:
            print(f"[prepare_data] Unsupported format: {fname}. "
                  "Please upload .tar.gz, .tgz, or .zip.")
            continue

    missing = [f for f in expected if not os.path.isdir(f)]
    if missing:
        raise RuntimeError(
            f"[prepare_data] The archive did not contain the expected "
            f"folders. Missing: {missing}.\n"
            "Please ensure the archive root contains chan180/, chan395/, "
            "chan590/, each with a profiles/ subdirectory."
        )
    print("[prepare_data] DNS data folders extracted: "
          "chan180, chan395, chan590.")


# =============================================================================
# §1  DNS data loading
#
# The dataset provides, at each wall-normal station, the mean velocity U+
# and the four non-zero components of the Reynolds-stress tensor
# <u'^2>, <v'^2>, <w'^2>, <u'v'>. These are the minimal inputs required to
# reconstruct the joint distribution of (u',v',w') under the Gaussian
# surrogate adopted in §4.
# =============================================================================
def load_dns(Re_tau):
    """
    Load mean and Reynolds-stress profiles from the DNS dataset.

    Parameters
    ----------
    Re_tau : int
        Friction Reynolds number (must be one of RE_LIST).

    Returns
    -------
    dict
        Keys: y_plus, U_plus, uu, vv, ww, uv. All arrays have the same
        length; the wall point (y+ = 0) is excluded.
    """
    folder = os.path.join(f"chan{Re_tau}", "profiles")
    rs    = np.loadtxt(os.path.join(folder, f"chan{Re_tau}.reystress"),
                       comments="#")
    means = np.loadtxt(os.path.join(folder, f"chan{Re_tau}.means"),
                       comments="#")
    mask = rs[:, 1] > 0.0
    return {
        "y_plus": rs[mask, 1],
        "uu":     rs[mask, 2],
        "vv":     rs[mask, 3],
        "ww":     rs[mask, 4],
        "uv":     rs[mask, 5],
        "U_plus": means[mask, 2],
    }


# =============================================================================
# §2  The exact hyperbolic identity (paper, eqs. 3 and 6)
#
#     cosh(2 phi)   = 1 + 2 |v|^2 / U_ref^2        (pointwise)
#     <cosh(2 phi)> = 1 + 2 U^2 + 4 k              (averaged)
#
# Both are exact. The function below returns the right-hand side of the
# averaged identity.
# =============================================================================
def rhs_identity(U, k):
    """Right-hand side of the exact averaged identity, eq. (6)."""
    return 1.0 + 2.0 * U**2 + 4.0 * k


# =============================================================================
# §3  The four candidate formulas for <phi'^2>
#
# F1  linear in <u'^2>          : <u'^2> / (1 + U^2)
# F2  isotropic baseline 2k/3    : (2k/3)  / (1 + U^2)
# F3  small-fluctuation limit    : 2k / (1 + 2 U^2)
# F4  closed form under A1-A3    : 0.5 ln(1 + 4k / (1 + 2 U^2))
# =============================================================================
def F1_linear_u(U, uu):
    return uu / (1.0 + U**2)

def F2_isotropic(U, k):
    return (2.0 * k / 3.0) / (1.0 + U**2)

def F3_small_fluct(U, k):
    return 2.0 * k / (1.0 + 2.0 * U**2)

def F4_closed(U, k):
    arg = 1.0 + 4.0 * k / (1.0 + 2.0 * U**2)
    return 0.5 * np.log(np.maximum(arg, 1e-300))


# =============================================================================
# §4  Monte Carlo surrogate
#
# We sample (u',v',w') from a zero-mean Gaussian with covariance matching
# the DNS Reynolds stresses, form phi = arcsinh(|v|), and compute the
# fluctuations phi' = phi - <phi>. Three statistics are returned:
#     <phi'^2>       : the target quantity
#     <cosh 2 phi'>  : appears in the exact decomposition, eq. (8)
#     <sinh 2 phi'>  : the skewness diagnostic of A2
# =============================================================================
def mc_surrogate(U, uu, vv, ww, uv=0.0, n=MC_SAMPLES, seed=RNG_SEED):
    """
    Monte Carlo surrogate for <phi'^2>, <cosh 2 phi'>, <sinh 2 phi'>.

    Returns
    -------
    dict with keys phi_mean, phi_sq, cosh, sinh.
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[uu, uv, 0.0],
                    [uv, vv, 0.0],
                    [0.0, 0.0, ww]])
    eig_min = float(np.min(np.linalg.eigvalsh(cov)))
    if eig_min < 1e-15:
        cov += (-eig_min + 1e-15) * np.eye(3)

    samples = rng.multivariate_normal([0.0, 0.0, 0.0], cov, size=n)
    u_p, v_p, w_p = samples[:, 0], samples[:, 1], samples[:, 2]

    v_mag = np.sqrt((U + u_p)**2 + v_p**2 + w_p**2)
    phi   = np.arcsinh(v_mag)
    phi_m = float(np.mean(phi))
    phi_p = phi - phi_m

    return {
        "phi_mean": phi_m,
        "phi_sq":   float(np.mean(phi_p**2)),
        "cosh":     float(np.mean(np.cosh(2.0 * phi_p))),
        "sinh":     float(np.mean(np.sinh(2.0 * phi_p))),
    }


# =============================================================================
# §4.2  Pointwise analysis of the DNS profiles
# =============================================================================
def analyse_dns(Re_tau):
    """
    Evaluate every candidate formula and the Monte Carlo surrogate at each
    available DNS station, and store the diagnostic quantities of §5.
    """
    dns  = load_dns(Re_tau)
    rows = []
    for i in range(dns["y_plus"].size):
        yp = float(dns["y_plus"][i])
        if not (Y_PLUS_MIN < yp < Y_PLUS_MAX):
            continue
        U  = float(dns["U_plus"][i])
        uu = float(dns["uu"][i])
        vv = float(dns["vv"][i])
        ww = float(dns["ww"][i])
        uv = float(dns["uv"][i])
        k  = 0.5 * (uu + vv + ww)

        mc = mc_surrogate(U, uu, vv, ww, uv)

        delta_phi  = mc["phi_mean"] - np.arcsinh(U)
        sinh_ratio = (mc["sinh"] / mc["cosh"]
                      if mc["cosh"] != 0.0 else 0.0)
        cosh_ratio = (mc["cosh"] / np.exp(2.0 * mc["phi_sq"])
                      if mc["phi_sq"] > 0 else 1.0)

        f4 = F4_closed(U, k)

        cosh2phi0_true = np.cosh(2.0 * mc["phi_mean"])
        cosh_true_noA2 = rhs_identity(U, k) / cosh2phi0_true
        phi_sq_A1      = 0.5 * np.log(max(cosh_true_noA2, 1e-300))

        sinh2phi0_true = np.sinh(2.0 * mc["phi_mean"])
        cosh_true_full = ((rhs_identity(U, k)
                           - sinh2phi0_true * mc["sinh"])
                          / cosh2phi0_true)
        phi_sq_A1A2    = 0.5 * np.log(max(cosh_true_full, 1e-300))

        phi_sq_mc = mc["phi_sq"]
        gap = phi_sq_mc - f4
        cA1 = ((phi_sq_A1 - f4) / gap) if abs(gap) > 1e-12 else 0.0
        cA2 = ((phi_sq_A1A2 - phi_sq_A1) / gap) if abs(gap) > 1e-12 else 0.0
        cA3 = ((phi_sq_mc - phi_sq_A1A2) / gap) if abs(gap) > 1e-12 else 0.0

        rows.append({
            "Re":             Re_tau,
            "y_plus":         yp,
            "U":              U,
            "k":              k,
            "uu":             uu,
            "vv":             vv,
            "ww":             ww,
            "uv":             uv,
            "phi_sq_mc":      phi_sq_mc,
            "cosh_mc":        mc["cosh"],
            "sinh_mc":        mc["sinh"],
            "cosh_rhs":       rhs_identity(U, k),
            "F1":             F1_linear_u(U, uu),
            "F2":             F2_isotropic(U, k),
            "F3":             F3_small_fluct(U, k),
            "F4":             f4,
            "phi_sq_A1":      phi_sq_A1,
            "phi_sq_A1A2":    phi_sq_A1A2,
            "delta_phi":      delta_phi,
            "sinh_ratio":     sinh_ratio,
            "cosh_ratio":     cosh_ratio,
            "gap_closure_A1": cA1,
            "gap_closure_A2": cA2,
            "gap_closure_A3": cA3,
        })
    return rows


# =============================================================================
# §6  Controlled information-barrier experiment
#
# At fixed (U, k), the state is determined by the anisotropy ratio
# A = (<v'^2> + <w'^2>) / <u'^2>, A in [0, 2]:
#     A = 0 : purely streamwise fluctuations
#     A = 2 : isotropic fluctuations
# Because any function f(U,k) assigns the same value to every point of this
# set, its pointwise error is at least half the spread (triangle inequality).
# =============================================================================
def barrier(U=8.0, k=4.0, n_A=21, A_max=2.0, n=MC_SAMPLES):
    """
    Controlled information-barrier experiment.
    """
    total  = 2.0 * k
    A_vals = np.linspace(0.0, A_max, n_A)
    rows   = []
    for A in A_vals:
        uu         = total / (1.0 + A) if A > 0 else total
        vv_plus_ww = total - uu
        vv         = vv_plus_ww / 2.0
        ww         = vv_plus_ww / 2.0
        mc = mc_surrogate(U, uu, vv, ww, uv=0.0, n=n, seed=RNG_SEED)
        rows.append({
            "A":      float(A),
            "uu":     float(uu),
            "vv":     float(vv),
            "ww":     float(ww),
            "phi_sq": mc["phi_sq"],
        })
    vals     = [r["phi_sq"] for r in rows]
    spread   = max(vals) - min(vals)
    bound    = 0.5 * spread
    rel      = spread / min(vals) if min(vals) > 0 else None
    return {
        "U":          U,
        "k":          k,
        "rows":       rows,
        "min":        float(min(vals)),
        "max":        float(max(vals)),
        "spread":     float(spread),
        "bound":      float(bound),
        "rel_spread": float(rel) if rel is not None else None,
    }


# =============================================================================
# Figure generation
# =============================================================================
def fig_phi_variance(rows, path):
    """Fig. 1: comparison of F1-F4 against the Monte Carlo surrogate."""
    fig, axes = plt.subplots(1, len(RE_LIST),
                             figsize=(5.5 * len(RE_LIST), 4.2),
                             sharey=False)
    if len(RE_LIST) == 1:
        axes = [axes]
    for ax, Re in zip(axes, RE_LIST):
        sub = [r for r in rows if r["Re"] == Re]
        if not sub:
            continue
        yp = [r["y_plus"] for r in sub]
        ax.loglog(yp, [r["phi_sq_mc"] for r in sub], "k-",
                  label="MC surrogate", lw=2)
        ax.loglog(yp, [r["F1"] for r in sub], "b--",
                  label=r"$F_1$ (linear in $\langle u^{\prime 2}\rangle$)")
        ax.loglog(yp, [r["F2"] for r in sub], "g:",
                  label=r"$F_2$ (isotropic $2k/3$)")
        ax.loglog(yp, [r["F4"] for r in sub], "r-.",
                  label=r"$F_4$ (closed form)")
        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\langle\phi^{\prime 2}\rangle$")
        ax.set_title(rf"$Re_\tau = {Re}$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_assumptions(rows, path):
    """Fig. 2: quantitative failure of A1, A2, A3 in the buffer layer."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for Re in RE_LIST:
        sub = [r for r in rows if r["Re"] == Re]
        if not sub:
            continue
        yp = [r["y_plus"] for r in sub]
        axes[0].plot(yp, [r["delta_phi"] for r in sub],
                     "o-", ms=3, label=rf"$Re={Re}$")
        axes[1].plot(yp, [r["sinh_ratio"] for r in sub],
                     "o-", ms=3, label=rf"$Re={Re}$")
        axes[2].plot(yp, [r["cosh_ratio"] for r in sub],
                     "o-", ms=3, label=rf"$Re={Re}$")
    titles = ["A1: mean shift",
              "A2: skewness ratio",
              "A3: Gaussianity ratio"]
    ylabels = [r"$\langle\phi\rangle - \operatorname{arcsinh}U$",
               r"$\langle\sinh 2\phi'\rangle/\langle\cosh 2\phi'\rangle$",
               r"$\langle\cosh 2\phi'\rangle / e^{2\langle\phi'^2\rangle}$"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.axhline(0 if ("mean" in title or "skew" in title) else 1,
                   color="k", ls="--", alpha=0.4)
        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_barrier(result, path):
    """Fig. 3: the controlled information-barrier experiment."""
    rows   = result["rows"]
    A_vals = [r["A"] for r in rows]
    phi_sq = [r["phi_sq"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(A_vals, phi_sq, "o-", color="C0",
            label=r"$\langle\phi^{\prime 2}\rangle_{\rm MC}$")
    ax.axhline(result["min"], color="gray", ls=":", alpha=0.6)
    ax.axhline(result["max"], color="gray", ls=":", alpha=0.6)
    ax.fill_between([0, max(A_vals)], result["min"], result["max"],
                    color="C1", alpha=0.15,
                    label=rf"spread = {result['spread']:.4f}")
    ax.axhline(result["min"] + result["bound"], color="C3", ls="--",
               label=rf"barrier bound $= {result['bound']:.4f}$")
    ax.set_xlabel(
        r"anisotropy ratio "
        r"$A=(\langle v^{\prime 2}\rangle+\langle w^{\prime 2}\rangle)"
        r"/\langle u^{\prime 2}\rangle$")
    ax.set_ylabel(r"$\langle\phi^{\prime 2}\rangle$")
    ax.set_title(rf"Information barrier at $U={result['U']}$, "
                 rf"$k={result['k']}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Results export
# =============================================================================
def export_csv(rows, path):
    keys = ["Re", "y_plus", "U", "k", "uu", "vv", "ww", "uv",
            "phi_sq_mc", "F1", "F2", "F3", "F4",
            "phi_sq_A1", "phi_sq_A1A2",
            "delta_phi", "sinh_ratio", "cosh_ratio",
            "gap_closure_A1", "gap_closure_A2", "gap_closure_A3"]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r[k]:.8g}" if isinstance(r[k], float) else str(r[k])
                for k in keys) + "\n")


def export_barrier_csv(result, path):
    with open(path, "w") as f:
        f.write("A,uu,vv,ww,phi_sq\n")
        for r in result["rows"]:
            f.write(f"{r['A']:.6g},{r['uu']:.6g},{r['vv']:.6g},"
                    f"{r['ww']:.6g},{r['phi_sq']:.8g}\n")


# =============================================================================
# Main driver
# =============================================================================
def main():
    print("=" * 78)
    print(" Information barrier of closed-form wall functions")
    print(" Supplementary material")
    print("=" * 78)

    # ---- §0  Data preparation ---------------------------------------------
    prepare_data()

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ---- §4  DNS comparison ----------------------------------------------
    all_rows = []
    for Re in RE_LIST:
        print(f"\n--- DNS comparison, Re_tau = {Re} ---")
        try:
            rows = analyse_dns(Re)
        except FileNotFoundError:
            print(f"    DNS folder chan{Re}/ not found; skipping.")
            continue
        all_rows.extend(rows)
        print(f"    {len(rows)} stations processed.")

    if all_rows:
        export_csv(all_rows, "results/dns_comparison.csv")
        fig_phi_variance(all_rows, "figures/fig1_phi_variance.pdf")
        fig_assumptions(all_rows, "figures/fig2_assumptions.pdf")

    # ---- §6  Information barrier ------------------------------------------
    print("\n--- Controlled information-barrier experiment ---")
    res = barrier(U=8.0, k=4.0)
    print(f"    U={res['U']}, k={res['k']}")
    print(f"    <phi'^2> range : [{res['min']:.4f}, {res['max']:.4f}]")
    print(f"    spread         :  {res['spread']:.4f}")
    print(f"    barrier bound  :  {res['bound']:.4f}")
    if res["rel_spread"] is not None:
        print(f"    relative spread:  {res['rel_spread']*100:.1f}%")
    export_barrier_csv(res, "results/barrier.csv")
    fig_barrier(res, "figures/fig3_barrier.pdf")

    # ---- Summary ----------------------------------------------------------
    summary = {
        "RE_LIST":    list(RE_LIST),
        "MC_SAMPLES": MC_SAMPLES,
        "RNG_SEED":   RNG_SEED,
        "barrier":    {k: res[k] for k in
                       ("U", "k", "min", "max", "spread",
                        "bound", "rel_spread")},
        "n_dns_rows": len(all_rows),
    }
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print(" Outputs:")
    print("   figures/fig1_phi_variance.pdf")
    print("   figures/fig2_assumptions.pdf")
    print("   figures/fig3_barrier.pdf")
    print("   results/dns_comparison.csv")
    print("   results/barrier.csv")
    print("   results/summary.json")
    print("=" * 78)


if __name__ == "__main__":
    main()