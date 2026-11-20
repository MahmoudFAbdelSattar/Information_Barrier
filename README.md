# Information Barrier of Algebraic Wall Functions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the supplementary material and Python code for reproducing the numerical results, tables, and figures presented in the paper:

> **"The Information Barrier of Algebraic Wall Functions: Why \(U\) and \(k\) Cannot Determine the Arcsinh-Transformed Velocity Variance"**
> *Mahmoud F. Abdel-Sattar* 
> *Submitted to Journal of Fluid Mechanics (Rapids)*

## 📖 Overview of the Theory

In turbulence modeling, a pervasive assumption is that one can construct a closed-form algebraic wall function \(f(U,k)\) to predict the variance of the arcsinh-transformed velocity, \(\langle\phi'^2\rangle\), using solely the mean velocity \(U\) and the turbulent kinetic energy \(k\). 

This code proves that this assumption is mathematically impossible. It demonstrates the existence of an **"Information Barrier"**: because \(k\) constrains only the trace of the Reynolds stress tensor, it lacks the directional information (anisotropy) required to determine individual variance components. 

Using a controlled Gaussian Monte Carlo surrogate parameterized by DNS data, the code quantifies this bound. It shows that at a fixed \((U, k)\) in the buffer layer, the true variance fluctuates by a factor of 4.7 depending on the anisotropy ratio \(A\). By the triangle inequality, this enforces a strict minimax lower bound, translating to a relative error strictly greater than **60%** for *any* arbitrary algebraic closure \(f(U,k)\).

## 📂 Repository Structure

```text
.
├── supplementary.py        # Main script to reproduce all results
├── README.md               # This file
├── figures/                # Output directory (generated upon run)
├── results/                # Output directory for CSV/JSON (generated upon run)
├── chan180/                # DNS Data folder (Re_tau = 180)
├── chan395/                # DNS Data folder (Re_tau = 395)
└── chan590/                # DNS Data folder (Re_tau = 590)
```

⚙️ Requirements

Ensure you have Python 3.8 or higher installed. You can install the required dependencies via pip:

```bash
pip install numpy scipy matplotlib
```

🚀 Usage

1. Data Preparation

The code requires the channel-flow DNS data of Moser, Kim & Mansour (1999) for Re_\tau = 180, 395, and 590.

· Local Run: Ensure the folders chan180, chan395, and chan590 (each containing a profiles/ subdirectory) are placed in the root directory.
· Google Colab: If running in Colab without the data, the script will automatically prompt you to upload a compressed archive (chandata.tar.gz or chandata.zip) containing the folders.

2. Run the Code

Execute the main script from your terminal:

```bash
python supplementary.py
```

3. Outputs

Upon successful execution, the script generates the following files:

· figures/fig1_phi_variance.pdf: Comparison of closed-form formulas against the Monte Carlo surrogate across the boundary layer.
· figures/fig2_assumptions.pdf: Quantitative failure of the mean shift, skewness, and Gaussianity assumptions.
· figures/fig3_barrier.pdf: The Information Barrier at fixed (U=8.0, k=4.0), illustrating the massive structural spread (\Delta = 0.1480).
· results/dns_comparison.csv: Full table of DNS comparisons and error diagnostics.
· results/barrier.csv: Raw data for the information barrier spread.
· results/summary.json: Summary of the Monte Carlo parameters and barrier bounds.

📝 Citation

If you use this code or find these results useful in your research, please cite the corresponding paper:

```bibtex
@article{AbdelSattar2026,
  title={The Information Barrier of Algebraic Wall Functions: Why \(U\) and \(k\) Cannot Determine the Arcsinh-Transformed Velocity Variance},
  author={Abdel-Sattar, Mahmoud F.},
  journal={Journal of Fluid Mechanics (Rapids)},
  year={2026},
  note={Under Review}
}
```

📧 Contact

For any questions, issues, or collaborations, please contact:
Mahmoud F. Abdel-Sattar
Department of Astronomy and Meteorology, Faculty of Science, Al-Azhar University, Cairo, Egypt.
Email: m.f.abdel-sattar@azhar.edu.eg

⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.
