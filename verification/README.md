# Independent verification

Two independent checks of the analysis, each written from the manuscript text alone, without access to the code in
`figure_code/`, `tables/` or `derivations/` (each has its own model library).

* `independent_check_1/`: all equations, identities, limiting values and quoted numbers of the manuscript up to
  Sec. VIII and Appendix A (sympy scripts `s*.py`, numerical scripts `n*.py`, model library `rlib.py`, logs `*.log`).
* `independent_check_2/`: the results added later: the necessary-and-sufficient condition (Sec. III C), flows near
  shear-free flows (`i1_*`), the lower bound for the rate-controlled protocol and its proof in Appendix B (`i2_*`),
  power-law and Rouse spectra (`i3_*`), the next-order structure and the fifth-order fluid (`i4_*`).

Discrepancies found by these checks were corrected in the manuscript; no equation was found in error.
Scripts run from their own folder, e.g. `cd independent_check_1 && python3 n1_class.py`.
