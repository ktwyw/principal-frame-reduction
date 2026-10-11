# Figure code

Python code that regenerates every figure in the manuscript and checks every number the text quotes from them. The manuscript is

> Y. Wang, *Principal-frame reduction of steady viscoelastic flows: generalizing the equivalence of shear and planar extension* (prepared for the Journal of Rheology).

## Requirements

- Python ≥ 3.10.
- `numpy`, `scipy`, `matplotlib`, `openpyxl` (the last is needed for Fig. 7 only).
- Optional: `sympy`, `mpmath`, for the derivation scripts in `../open_questions` and `../independent_check`.

## Quick start

```
python3 run_all.py /path/to/PIB_freqsweep_flowcurve.xlsx    # everything, about 20 min on 2 cores
python3 run_all.py --plot                                   # re-plot from the stored data in data/
```

Outputs:
- `figures/fig*.pdf`: vector figures with embedded TrueType fonts.
- `data/fig*.json`: the numbers behind each figure.
- `CHECKS.md`: every number the manuscript quotes from a figure, recomputed, with a verdict.

Each script also runs on its own, e.g. `python3 fig3_spectra.py`. Add `--plot` to redraw from `data/` without recomputing.

## Files

| File | Content | Runtime |
|---|---|---|
| `common.py` | Plot style; kinematics; single-tensor models (UCM, Johnson-Segalman, linear and exponential PTT, Giesekus, FENE-P, Rolie-Poly); steady-state solver for any homogeneous flow; the King–McKinley composite and the principal-frame projection; closed-form single-mode material functions for multimode fluids; K-BKZ quadrature; the `Checks` class. Running `python3 common.py` self-tests the closed forms against the full-tensor solver. | 1 min |
| `fig1_collapse.py` | **Fig. 1**: Giesekus and Rolie-Poly in planar flows of four flow types, projected onto the stress eigenframe, against independently computed planar extension. | ~10 min |
| `fig2_protocol.py` | **Fig. 2**: the rate-controlled shear protocol (Eq. 19) against start-up of planar extension; the shear strain required, against exponential shear. | 1 min |
| `fig3_spectra.py` | **Fig. 3**: Rouse and Zimm chains, exact against composite; the scaled slow-flow defect (Eq. 34) for four nonlinear two-mode fluids and the Rouse chain. | 2 min |
| `fig4_outside.py` | **Fig. 4**: the ratio composite/planar viscosity for White–Metzner, K-BKZ, multimode PTT, Oldroyd-B and Rouse fluids; the multimode-PTT scans; the high-rate Cauchy–Schwarz factor. | 5 min |
| `fene_bd.py` | Brownian dynamics engine for FENE dumbbells without closure (semi-implicit predictor-corrector), used for Fig. 5 and Table 5. | hours |
| `fig5_fene.py` | **Fig. 5**: FENE-P (computed) against FENE Brownian dynamics (stored in `data/fene_bd_records.json`; regenerate with `--recompute`). | 1 min |
| `fig6_nemd.py` | **Fig. 6 and Table 4**: the NEMD data of Daivis, Matin and Todd (2003, 2007), the composite built from them, and Maxwell modes with the Rouse spectrum. | seconds |
| `fig7_pib.py` | **Fig. 7**: polyisobutylene solution. Relaxation spectra, composite with solvent subtraction, multimode PTT and Giesekus fits, single-mode Rolie-Poly. Needs the public data file. | 5 min |
| `run_all.py` | Runs everything and writes `CHECKS.md`. | |

## Data sources

- **Fig. 6.** Constants from Daivis, Matin & Todd, *J. Non-Newtonian Fluid Mech.* 147, 35 (2007), Tables 2–3. Curves digitised from Daivis, Matin & Todd, *J. Non-Newtonian Fluid Mech.* 111, 1 (2003), Figs. 1, 2 and 4.
  - The digitised values are typed into `fig6_nemd.py`.
  - Strain rates are exact.
  - Ordinates carry about 3% digitisation error (about 8% where markers overlap).
- **Fig. 7.** N. King, figshare (2026), doi:10.6084/m9.figshare.32005782, licence CC BY 4.0. Pass the path of `PIB_freqsweep_flowcurve.xlsx`.
- **Fig. 5.** The Brownian dynamics output is stored in `data/fene_bd_records.json`: 30,000 dumbbells, b = 50 (plus b = 20 and 100), and a time-step and a seed check at Wi = 10.

## Conventions

- Velocity gradient: L_ij = ∂v_i/∂x_j. Simple shear: L = γ̇ e₁e₂. Planar extension: L = ε̇ diag(1, −1, 0).
- Single-mode models use the units η₀ = τ = G = 1. A mode (η_k, τ_k) of a multimode fluid is obtained by scaling rates by τ_k and stresses by η_k/τ_k.
- Steady states are found by integrating the full six-component tensor equation from rest (Radau, rtol 10⁻¹⁰) and polishing with a Newton solve. Shear-free reference states are always computed independently from rest, never from a mapped state.
