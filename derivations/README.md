# Derivations and checks

Symbolic (sympy) and numerical checks of the analytical results of the paper. Each script is standalone, runs from
this folder (`python3 <script>.py`), prints its results and writes a `.json` with the same results.
`run_all.sh` runs everything (about 1 h on 2 cores). `lib.py` contains the steady-state solver and models used by
several scripts (the same models as `../figure_code/common.py`, in an earlier, more compact form).

| Script | What it establishes | Manuscript |
|---|---|---|
| `th2_2.py` | Realizability lemma: at a fixed principal state, any off-diagonal rate h is realized by a suitable vorticity (Giesekus, d1*tau = 0.8: W'12*tau = 0.34, 1.13, 3.40) and reached from rest | Sec. III C |
| `oq_nearsf.py` | Reduction exact to first order near shear-free flows (two-mode Maxwell fluid: mismatch -> -0.07 delta^2) | Sec. III D |
| `other_ext.py` | Shear + stretch along the vorticity axis mapped onto uniaxial and equibiaxial extension (agreement < 1e-13) | Sec. III D |
| `th2_3.py` | Short-time expansion of the rate-controlled protocol for a Maxwell spectrum: gdot(t) and the error 1 - eps0^2 t^4 Var_G(1/tau)/120 | Eq. (21) |
| `th2_5.py` | (a) the same series to higher order, checked against numerical integration; (b) K-BKZ O(eps^4) formula against direct quadrature | Eqs. (21), (38) |
| `th2_7.py`, `oq_bound.py`, `oq1.py`, `oq2.py` (a), `oq3.py`, `oq5.py` | Lower bound for the protocol: exact identity (24), ordering of the modes, monotonic shear rate, gamma <= 2 sinh(eps0 t), on random spectra (2-10 modes, up to 4 decades, rates on both sides of 1/2 tau_max) | Eqs. (23)-(26), Appendix B |
| `oq_powerlaw.py`, `oq_rouse.py`, `oq2.py` (b), `oq4.py` | Power-law relaxation: universal error function of Hencky strain, coefficient 2 alpha (1 - alpha)/[(3 - alpha)(2 - alpha)^3]; 200-mode Rouse spectrum | Eqs. (27)-(28) |
| `oq_fifth.py`, `oq2.py` (c) | O(eps^4) composite coefficient in terms of shear expansion coefficients; spanning set of fifth-order-fluid terms; consistency with K-BKZ | Eq. (36) |
| `th2_4.py` | Separable K-BKZ fluid with arbitrary memory and damping: eta_P1 - eta_[P1] through O(eps^4), and the P2 analogue | Eq. (38), Appendix A |
| `oq2.py` (d) | Dumbbell Fokker-Planck: generalized Boltzmann ansatz is exact for Hookean springs, not for FENE | Sec. VI |
| `check9.py` | Multimode linear-PTT scans quoted in Sec. V D; start-up of shear vs planar extension (51 % at equal Hencky strain) | Secs. IV, V D |
| `spectra.py` | Moments of the Rouse and Zimm spectra (7/10, 0.524) | Sec. V C |
