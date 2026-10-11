# Principal-frame reduction of steady viscoelastic flows: code and data

This repository holds the code and data for

> Y. Wang, *Principal-frame reduction of steady viscoelastic flows: generalizing the equivalence of shear and planar extension*, submitted to the Journal of Rheology (2026).

The paper builds on the composite proposed by N. King and G. H. McKinley, Phys. Rev. Lett. **137**, 144003 (2026). Their composite projects steady-shear data onto the principal axes of the stress to estimate the planar extensional viscosity.

The repository contains:
- the code that produces every figure and table in the paper;
- every number the paper quotes from them;
- the symbolic and numerical checks behind every derivation;
- two independent re-derivations of the analytical results.

## Quick start

```
pip install -r requirements.txt

cd figure_code
python3 run_all.py ../data/external/pib/PIB_freqsweep_flowcurve.xlsx   # Figs. 1-7, about 10-20 min
cat CHECKS.md                                                           # 124 numbers quoted in the paper, recomputed

cd ../tables
python3 make_tables.py        # Tables 2-6, compared line by line with the manuscript
python3 class_table.py        # Table 1 (about 10 min; needs extend.json, written by extend.py)

cd ../derivations
sh run_all.sh                 # every derivation and check script, about 1 h
```

The code was tested with Python 3.13, numpy 2.5, scipy 1.18, sympy 1.14, mpmath 1.3, matplotlib 3.11 and openpyxl 3.1. Any reasonably recent versions should work.

## Layout

| Folder | Contents |
|---|---|
| `figure_code/` | Self-contained, documented code for Figs. 1–7. Includes `common.py` (constitutive models, steady-state solver, composite, multimode functions, K-BKZ quadrature) and `fene_bd.py` (Brownian dynamics of FENE dumbbells). `data/` holds the numbers behind each figure, `figures/` the vector PDFs, and `CHECKS.md` the verification report. See `figure_code/README.md`. |
| `tables/` | Code and data for Tables 1–6. `output/` is regenerated; `manuscript/` holds the tables as typeset in the paper, for comparison. |
| `derivations/` | Symbolic (sympy) and numerical checks of the analytical results. See `derivations/README.md` for the map to the equations of the paper. |
| `verification/` | Two independent re-derivations, written from the manuscript text alone without access to the code in the other folders. |
| `data/external/pib/` | Third-party rheological data for the polyisobutylene solution, CC BY 4.0. |
| `data/nemd/` | Raw output of the digitization of published molecular dynamics figures, and the digitization script. |

## Where each result comes from

| Manuscript item | Script | Output |
|---|---|---|
| Fig. 1, collapse of planar flows | `figure_code/fig1_collapse.py` | `figure_code/data/fig1.json` |
| Fig. 2, rate-controlled shear protocol | `figure_code/fig2_protocol.py` | `fig2.json` |
| Fig. 3, Rouse/Zimm spectra; slow-flow defect, Eq. (34) | `figure_code/fig3_spectra.py` | `fig3.json` |
| Fig. 4, fluids outside the class; multimode PTT scans | `figure_code/fig4_outside.py` | `fig4.json` |
| Fig. 5 and Table 5, FENE dumbbells without closure | `figure_code/fene_bd.py`, `fig5_fene.py`, `tables/make_tables.py` | `figure_code/data/fene_bd_records.json` |
| Fig. 6 and Table 4, NEMD data of Daivis, Matin & Todd | `figure_code/fig6_nemd.py`, `tables/make_tables.py` | `fig6.json` |
| Fig. 7, polyisobutylene solution | `figure_code/fig7_pib.py` | `fig7.json` |
| Table 1, class membership | `tables/extend.py`, `tables/class_table.py` | `tables/extend.json` |
| Table 2, converse in 3D flows | `tables/converse_3d.py` | `tables/theory2_6.json` |
| Table 3, 3D flows | `tables/make_tables.py` | `tables/output/flows_table.tex` |
| Table 6, coaxiality of stress and second moment | `tables/fene_bd3.py` | `tables/fene_bd3.json` |
| Sec. III C, realizability lemma (Giesekus numbers) | `derivations/th2_2.py` | `theory2_2.json` |
| Sec. III D, flows near shear-free flows | `derivations/oq_nearsf.py` | `oq_nearsf.json` |
| Sec. III D, uniaxial and biaxial extension | `derivations/other_ext.py` | `other_ext.json` |
| Sec. IV A, short-time error, Eq. (21) | `derivations/th2_3.py`, `th2_5.py` | `theory2_3.json` |
| Sec. IV A, lower bound, Eqs. (23)-(24) and Appendix B | `derivations/oq_bound.py`, `oq1.py`, `oq2.py`, `oq3.py`, `oq5.py`, `th2_7.py` | `oq_bound.json`, `oq2.json` |
| Sec. IV A, Rouse-like spectra, Eq. (28) | `derivations/oq_powerlaw.py`, `oq_rouse.py`, `oq4.py` | `oq_powerlaw.json`, `oq_rouse.json` |
| Sec. IV, start-up comparison (51%) | `derivations/check9.py` | `check9.json` |
| Sec. V B, next order; fifth-order fluid, Eq. (36) | `derivations/oq_fifth.py`, `oq2.py` | `oq_fifth.json` |
| Sec. V D and Appendix A, K-BKZ to O(ε̇⁴), Eq. (38) | `derivations/th2_4.py`, `th2_5.py` | `theory2_4.json` |
| Sec. VI, dumbbell distribution criterion | `derivations/oq2.py` (part d) | printed |

Equation numbers refer to the submitted version.

## Licences

- **Code:** MIT licence (`LICENSE`).
- **Data generated in this work:** the JSON, PDF and TeX outputs are released under CC BY 4.0 (`DATA_LICENSE.md`).
- **Third-party data:** the conditions of the original sources apply (see `DATA_LICENSE.md`).

## Citation

If you use this code or data, please cite the paper above and this repository (`CITATION.cff`).

## Contact

Yanwei Wang, Nazarbayev University: yanwei.wang@nu.edu.kz, ORCID [0000-0002-8488-9833](https://orcid.org/0000-0002-8488-9833).
