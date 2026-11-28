The Information Barrier of Algebraic Wall Functions

Computational Supplementary Material

This repository contains the computational framework accompanying the research manuscript:

«The Information Barrier of Algebraic Wall Functions: Why U and k Cannot Determine the Arcsinh-Transformed Velocity Variance»

The project investigates a fundamental information limitation of algebraic turbulence closures that attempt to determine the variance of an arcsinh-transformed velocity magnitude using only the mean streamwise velocity (U) and turbulent kinetic energy (k).

The computational experiments quantify this limitation within controlled Gaussian velocity ensembles and examine its implications using DNS-parameterized Gaussian surrogate data.

---

1. Scientific Objective

The central quantity considered in this study is

[
\phi = \operatorname{arcsinh}(|\mathbf{v}|),
]

where the instantaneous velocity vector is

[
\mathbf{v}=(U+u',v',w').
]

The target statistic is

[
g=\operatorname{Var}(\phi).
]

The central question is whether (g) can be uniquely determined by the two scalar quantities (U) and (k).

The manuscript establishes an information barrier for any deterministic algebraic closure of the form

[
\widehat{g}=f(U,k).
]

If two admissible velocity states share the same values of (U) and (k), but produce different values of (g), no such closure can reproduce both states exactly.

For two admissible states with target values (g_1) and (g_2), the minimax error satisfies

[
\max\left(
|f(U,k)-g_1|,
|f(U,k)-g_2|
\right)
\geq \frac{|g_1-g_2|}{2}.
]

This lower bound is independent of the particular algebraic form of the closure.

---

2. Computational Framework

The numerical investigation contains two conceptually distinct components.

2.1 Controlled Gaussian anisotropy experiment

A controlled family of Gaussian velocity fluctuations is constructed while keeping (U) and (k) fixed.

The anisotropy ratio is defined as

[
A=
\frac{\langle v'^2\rangle+\langle w'^2\rangle}
{\langle u'^2\rangle}.
]

Under the controlled assumptions of zero cross-correlations and equal transverse variances,

[
\Sigma(A)=
\begin{pmatrix}
\dfrac{2k}{1+A} & 0 & 0\[6pt]
0 & \dfrac{Ak}{1+A} & 0\[6pt]
0 & 0 & \dfrac{Ak}{1+A}
\end{pmatrix}.
]

Consequently,

[
\langle u'^2\rangle=\frac{2k}{1+A},
]

and

[
\langle v'^2\rangle

\langle w'^2\rangle

\frac{Ak}{1+A}.
]

The principal controlled experiment uses

[
U=8,\qquad k=4,
]

and examines the anisotropy range

[
A\in[0,2].
]

An extended sweep explores

[
A\in[0,10].
]

This construction isolates the effect of changing the distribution of turbulent kinetic energy among velocity components while preserving the same mean velocity and total turbulent kinetic energy.

Important: This is a controlled Gaussian subfamily, not a representation of the entire realizable Reynolds-stress state space.

2.2 DNS-parameterized Gaussian surrogate

A separate computational component uses turbulence statistics obtained from DNS channel-flow data.

The analysis considers the friction Reynolds numbers

[
Re_\tau=180,\quad 395,\quad 590.
]

The available mean-velocity and Reynolds-stress statistics are used to construct covariance matrices for Gaussian surrogate ensembles.

The resulting Monte Carlo samples are used to evaluate the transformed velocity statistics.

This procedure is a DNS-parameterized Gaussian surrogate analysis, not direct sampling of instantaneous DNS velocity fields. It preserves the prescribed second-order covariance information but does not reproduce the full non-Gaussian probability distribution of the original turbulent velocity fluctuations.

---

3. Exact Statistical Identity

The computational framework is connected to the exact identity

[
\left\langle\cosh(2\phi)\right\rangle

1+2U^2+4k
]

when velocities are expressed in wall units and the velocity fluctuations have zero mean.

In dimensional variables,

[
\left\langle\cosh(2\phi)\right\rangle

1+\frac{2U^2+4k}{u_\tau^2}.
]

This identity does not require Gaussian statistics or isotropy.

It provides an exact constraint on the transformed velocity distribution, while the variance of (\phi) remains sensitive to additional statistical information.

---

4. Numerical Experiments and Diagnostics

The computational study includes the following analyses:

- Evaluation of the arcsinh-transformed velocity variance.
- Controlled variation of the anisotropy ratio at fixed (U) and (k).
- Estimation of the spread in the target statistic across admissible states.
- Evaluation of the corresponding minimax lower bound.
- Extended anisotropy sweeps.
- Independent random-seed comparisons.
- Nested Monte Carlo sample-size convergence.
- Paired adjacent-increment diagnostics for the numerical anisotropy sweep.
- DNS-parameterized Gaussian surrogate comparisons.

Monte Carlo settings

The principal extended anisotropy sweep uses:

- 1001 anisotropy points.
- (A\in[0,10]).
- (\Delta A=0.01).
- (N=200{,}000) samples per state.
- Five independent random seeds.

The paired local-monotonicity diagnostics examine seeds 29 and 71 and account for multiple comparisons across 1000 adjacent intervals using Bonferroni correction.

The convergence analysis examines increasing sample sizes up to

[
N=1.6\times10^6.
]

---

5. Reported Numerical Findings

For the controlled experiment, the reported mean spreads and minimax bounds are:

Anisotropy range| Mean spread| Minimax bound
(A\in[0,2])| 0.1454| 0.0727
(A\in[0,10])| 0.1735| 0.0868

Extending the anisotropy range from ([0,2]) to ([0,10]) increases the mean spread by approximately 19.4%.

The paired adjacent-increment diagnostics detect no statistically significant upward increments in the tested numerical sweeps, including after Bonferroni correction.

These numerical observations do not constitute an analytical proof of continuous monotonicity of

[
g(A)=\operatorname{Var}(\phi)
]

at finite (U).

The distinction between numerical evidence and analytical proof is maintained throughout the study.

---

6. Reproducibility

The computational results depend on the following elements:

1. The supplied DNS channel-flow data.
2. The prescribed covariance construction.
3. The controlled Gaussian covariance family.
4. The Monte Carlo sample sizes.
5. The random seeds.
6. The anisotropy sampling grid.
7. The statistical post-processing and diagnostic procedures.

The DNS data preparation workflow requires the archive

"chandata.tar.gz"

where requested by the corresponding data-preparation script.

The archive should be placed in the location expected by that script before running the DNS-data preparation stage.

For reproducibility, retain the original random seeds, sample sizes, and anisotropy grid used in the reported experiments.

Execution instructions: The exact script entry points, command-line arguments, and dependency versions should be specified here after matching this documentation against the final repository contents.

---

7. Interpretation and Scope

The results should be interpreted within the assumptions of the corresponding numerical experiment.

In particular:

- The controlled Gaussian experiment demonstrates an information barrier within a specified admissible subfamily.
- The DNS-parameterized Gaussian surrogate uses DNS-derived statistical inputs but does not reproduce the complete instantaneous DNS probability distribution.
- The minimax theorem is independent of the Gaussian surrogate used in the numerical evaluation.
- The Reynolds-stress tensor supplies additional second-order information, but does not necessarily determine the complete probability distribution required to evaluate nonlinear transformed statistics.
- The observed finite-(U) behavior of the arcsinh-transformed variance should not be confused with a general analytical monotonicity theorem.

The analytical result concerning the variance of the squared velocity magnitude is also distinct from a proof of monotonicity of the arcsinh-transformed variance.

---

8. Repository Structure

The repository is organized around the computational scripts, input data preparation, numerical diagnostics, and generated results.

The final directory structure and script descriptions should be added here to reflect the actual files in the repository.

Suggested documentation categories:

- Main computational scripts.
- DNS data preparation.
- Controlled Gaussian experiments.
- Monte Carlo convergence analysis.
- Paired monotonicity diagnostics.
- Output tables and figures.

---

9. Scientific Reference

The computational material accompanies the manuscript:

The Information Barrier of Algebraic Wall Functions: Why U and k Cannot Determine the Arcsinh-Transformed Velocity Variance.

The repository is intended to support examination, reproduction, and independent verification of the numerical experiments reported in that work.

---

10. Contact

For questions concerning the computational methodology, numerical implementation, or scientific interpretation, please use the GitHub repository's issue-tracking facility or contact the corresponding author.
