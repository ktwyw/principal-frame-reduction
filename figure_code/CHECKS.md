# Numbers quoted in the manuscript, recomputed

124 of 124 checks pass.

Default tolerance: half a unit in the last printed digit; "<=" marks upper-bound statements.

| Figure | Quantity | Manuscript | Recomputed | Verdict |
|---|---|---|---|---|
| fig1 | Giesekus: collapse is exact (mismatch < 1e-9) | 0 | 7.5939e-14 | OK |
| fig1 | Rolie-Poly: collapse is exact (mismatch < 1e-9) | 0 | 1.5083e-12 | OK |
| fig2 | protocol = start-up of planar extension for the class (better than 1e-10) | 0 | 8.0467e-11 | OK |
| fig2 | shear strain at Hencky strain 6, linear PTT, eps0 tau = 2 | 36.4 | 36.359 | OK |
| fig2 | shear strain at Hencky strain 6, Giesekus, eps0 tau = 2 | 42.5 | 42.509 | OK |
| fig2 | shear strain at Hencky strain 6, FENE-P, eps0 tau = 2 | 62.1 | 62.102 | OK |
| fig2 | exponential shear at Hencky strain 6 | 403 | 403.43 | OK |
| fig2 | two-mode fluid, eps0 taubar = 0.2: shear/planar at t = 1 taubar | 0.999 | 0.99901 | OK |
| fig2 | two-mode fluid, eps0 taubar = 0.2: shear/planar at t = 3 taubar | 0.979 | 0.97873 | OK |
| fig2 | two-mode fluid, eps0 taubar = 0.2: shear/planar at t = 10 taubar | 0.86 | 0.86224 | OK |
| fig2 | two-mode fluid: steady limit of the ratio | 0.77 | 0.77232 | OK |
| fig2 | two-mode fluid: ratio at t = 50 taubar approaches the steady limit | 0.77 | 0.77251 | OK |
| fig2 | two-mode fluid: ratio decreases monotonically (1 = yes) | 1 | 1 | OK |
| fig3 | Rouse <tau>/tau_1 = pi^2/15 | 0.658 | 0.65797 | OK |
| fig3 | Rouse <tau>^2/<tau^2> | 0.70 | 0.7 | OK |
| fig3 | Zimm <tau>^2/<tau^2> | 0.524 | 0.52442 | OK |
| fig3 | Rouse: composite diverges at eps tau_1 | 0.76 | 0.75991 | OK |
| fig3 | Zimm: composite diverges at eps tau_1 | 1.09 | 1.0866 | OK |
| fig3 | Rouse eta_P1/eta0 at eps tau_1 = 0.4 | 8.35 | 8.3511 | OK |
| fig3 | Rouse eta_[P1]/eta0 at eps tau_1 = 0.4 | 5.53 | 5.5331 | OK |
| fig3 | linear PTT: scaled defect tends to 1 at the lowest rate | 1.0 | 1.0015 | OK |
| fig3 | Giesekus: scaled defect tends to 1 at the lowest rate | 1.0 | 1.0002 | OK |
| fig3 | exponential PTT: scaled defect tends to 1 at the lowest rate | 1.0 | 1.0015 | OK |
| fig3 | K-BKZ (PSM, a = 0.2): scaled defect tends to 1 at the lowest rate | 1.0 | 0.99188 | OK |
| fig3 | linear PTT: scaled P2 defect at eps taubar = 0.0100 | 0.945 | 0.94149 | OK |
| fig3 | exponential PTT: scaled P2 defect at eps taubar = 0.0100 | 0.945 | 0.94149 | OK |
| fig3 | Giesekus: scaled P2 defect at eps taubar = 0.0100 | 0.945 | 0.95227 | OK |
| fig3 | two-mode spectrum: eta_P1 at eps taubar = 0.25, single mode | 5.33 | 5.3333 | OK |
| fig3 | two-mode spectrum: eta_P1 at eps taubar = 0.25, two modes | 12.55 | 12.547 | OK |
| fig4 | White-Metzner at tau0 gdot = 5: composite / eta0 | 1.13 | 1.1296 | OK |
| fig4 | White-Metzner at the same eps_eff: eta_P1 / eta0 | 2.23 | 2.2304 | OK |
| fig4 | K-BKZ a = 0.05: largest overshoot of the composite (76%) | 1.76 | 1.7586 | OK |
| fig4 | K-BKZ a = 0.05: rate of the largest overshoot, eps_eff tau | 1.3 | 1.3303 | OK |
| fig4 | PTT, unequal eps: largest overshoot (59%) | 1.59 | 1.5906 | OK |
| fig4 | PTT, common eps: composite never exceeds the planar viscosity | 1.000 | 0.99994 | OK |
| fig4 | PTT, common eps: high-rate Cauchy-Schwarz factor | 0.890 | 0.8905 | OK |
| fig4 | PTT, common eps: numerical ratio at gdot taubar = 1e6 | 0.890 | 0.89048 | OK |
| fig4 | scan, eps >= 0.03: smallest ratio | 0.28 | 0.28211 | OK |
| fig4 | scan, eps >= 0.03: largest ratio | 1.00 | 1 | OK |
| fig4 | scan, eps = 0.003: range low | 0.06 | 0.058634 | OK |
| fig4 | scan, eps = 0.003: range high | 1.06 | 1.0608 | OK |
| fig4 | scan, eps = 0.001: range low | 0.02 | 0.021018 | OK |
| fig4 | scan, eps = 0.001: range high | 1.17 | 1.1747 | OK |
| fig5 | BD: composite/true at Wi = 2 | 0.87 | 0.87317 | OK |
| fig5 | BD: composite/true at Wi = 3 | 0.70 | 0.70119 | OK |
| fig5 | BD: composite/true at Wi = 5 | 0.52 | 0.52248 | OK |
| fig5 | BD: smallest ratio for Wi >= 10 (43%) | 0.43 | 0.42906 | OK |
| fig5 | BD: largest ratio for Wi >= 10 (47%) | 0.47 | 0.46806 | OK |
| fig5 | BD: ratio at Wi = 0.5 within 1% | 1.00 | 1.0023 | OK |
| fig5 | BD: ratio at Wi = 1 within 1% | 1.00 | 0.99225 | OK |
| fig5 | BD: Wi = 10, halved time step | 0.467 | 0.46702 | OK |
| fig5 | BD: Wi = 10, other seed | 0.467 | 0.46722 | OK |
| fig5 | BD: b = 20 | 0.48 | 0.4837 | OK |
| fig5 | BD: b = 100 | 0.46 | 0.45723 | OK |
| fig5 | FENE-P overestimates N1 at Wi = 3 (31%) | 0.31 | 0.30817 | OK |
| fig5 | FENE-P overestimates N1 at Wi = 100 (53%) | 0.53 | 0.53238 | OK |
| fig5 | FENE-P vs FENE planar viscosity for eps lambda >= 1.3: below 5% (|difference| - 0.025 within 0.025) | 0.025 | 0.019147 | OK |
| fig5 | FENE-P vs FENE planar viscosity: up to 60% near the transition | 0.6 | 0.58092 | OK |
| fig6 | Table 4: b2^2/(b1 b3) for N = 2 | 0.43 | 0.43499 | OK |
| fig6 | Table 4: b2^2/(b1 b3) for N = 4 | 0.52 | 0.51791 | OK |
| fig6 | Table 4: b2^2/(b1 b3) for N = 10 | 0.61 | 0.60808 | OK |
| fig6 | Table 4: b2^2/(b1 b3) for N = 20 | 0.77 | 0.76892 | OK |
| fig6 | Table 4: b2^2/(b1 b3) for N = 50 | 0.48 | 0.47577 | OK |
| fig6 | Table 4: b3 - b2^2/b1 > 0 for N = 2..50 (smallest value / its error) | 1 | 1 | OK |
| fig6 | measured / composite curvature of eta_P1, N = 50 (3.6 times) | 3.6 | 3.5878 | OK |
| fig6 | opposite sign for N = 10 (1 = yes) | 1 | 1 | OK |
| fig6 | eta_P2-based b3: b1 b3 < b2^2 for N = 20 by (standard errors) | 1.3 | 1.3211 | OK |
| fig6 | eta_P2-based b3: b1 b3 < b2^2 for N = 50 by (standard errors) | 2.3 | 2.3117 | OK |
| fig6 | N = 50: tau_1/taubar | 1.52 | 1.5182 | OK |
| fig6 | N = 20: tau_1/taubar | 1.51 | 1.5097 | OK |
| fig6 | N = 50: Rouse coil-stretch rate 1/2tau_1 | 4.9e-4 | 0.00048578 | OK |
| fig6 | N = 20: Rouse coil-stretch rate 1/2tau_1 | 3.3e-3 | 0.003344 | OK |
| fig6 | N = 50: maximum of eta_P1 / 4 eta0 (constant P_zz, first peak) | 3.6 | 3.6011 | OK |
| fig6 | N = 20: maximum of eta_P1 / 4 eta0 | 1.9 | 1.9071 | OK |
| fig6 | N = 50: largest N1/2s12 | 1.8 | 1.7747 | OK |
| fig6 | N = 20: largest N1/2s12 | 0.9 | 0.89492 | OK |
| fig6 | N = 50: composite within 8% of 4 eta0 up to eps_eff taubar ~ 1 | <= 0.08 | 0.070647 | OK |
| fig6 | N = 20: composite within 8% of 4 eta0 up to eps_eff taubar ~ 0.4 | <= 0.08 | 0.034895 | OK |
| fig6 | N = 50: ratio at eps_eff taubar ~ 2 | 0.25 | 0.2467 | OK |
| fig6 | N = 50: ratio at the highest rate, constant-volume data | 0.17 | 0.16714 | OK |
| fig6 | N = 50: ratio at the highest rate, constant-normal-pressure data | 0.24 | 0.2385 | OK |
| fig6 | abstract: factor four to six (1/ratio, constant volume) | 6 | 5.983 | OK |
| fig6 | abstract: factor four to six (1/ratio, constant P_zz) | 4 | 4.1928 | OK |
| fig6 | N = 20: smallest ratio | 0.34 | 0.33522 | OK |
| fig7 | chain-like spectrum exponent k | 1.90 | 1.899 | OK |
| fig7 | uncertainty of k | 0.02 | 0.024839 | OK |
| fig7 | slope of G' above 10 rad/s | 0.51 | 0.51459 | OK |
| fig7 | slope of G'' between 0.1 and 0.4 rad/s | 0.8 | 0.80992 | OK |
| fig7 | slope of G' between 0.1 and 0.4 rad/s, without the lowest point | 1.4 | 1.3873 | OK |
| fig7 | slope of G' between 0.1 and 0.4 rad/s, all points | 1.6 | 1.6341 | OK |
| fig7 | NNLS rms relative residuals: smallest | 0.05 | 0.053088 | OK |
| fig7 | NNLS rms relative residuals: largest | 0.07 | 0.068645 | OK |
| fig7 | eta0 of each spectrum | 3.8 | 3.8412 | OK |
| fig7 | eta0 of each spectrum | 3.8 | 3.7991 | OK |
| fig7 | eta0 of each spectrum | 3.8 | 3.8143 | OK |
| fig7 | eta0 of each spectrum | 3.8 | 3.8079 | OK |
| fig7 | eta0 of each spectrum | 3.8 | 3.8319 | OK |
| fig7 | <tau>: smallest over spectra | 1.3 | 1.2563 | OK |
| fig7 | <tau>: largest over spectra | 1.7 | 1.731 | OK |
| fig7 | <tau>^2/<tau^2>: smallest | 0.3 | 0.31067 | OK |
| fig7 | <tau>^2/<tau^2>: largest | 0.7 | 0.67718 | OK |
| fig7 | leading correction at 0.1 1/s as a fraction of 4 eta0: smallest (3%) | 0.03 | 0.030097 | OK |
| fig7 | leading correction at 0.1 1/s as a fraction of 4 eta0: largest (27%) | 0.27 | 0.26595 | OK |
| fig7 | Rouse-like coil-stretch rate 1/2tau_1 from the chain spectrum, low end | 0.25 | 0.25086 | OK |
| fig7 | solvent share of the shear stress at 63 1/s (3%) | 0.03 | 0.031029 | OK |
| fig7 | largest change of eta_[P1] by solvent subtraction (at most 3%) | 0.03 | 0.030927 | OK |
| fig7 | 4 eta0 [Pa s] | 15.3 | 15.196 | OK |
| fig7 | composite / 4 eta0 for eps_eff <= 1: smallest (10% above) | 1.10 | 1.0854 | OK |
| fig7 | composite / 4 eta0 for eps_eff <= 1: largest (25% above) | 1.25 | 1.2539 | OK |
| fig7 | composite at the highest rate [Pa s] | 70 | 69.92 | OK |
| fig7 | highest eps_eff [1/s] | 5.7 | 5.6609 | OK |
| fig7 | composite at 1 1/s [Pa s] | 19 | 19.116 | OK |
| fig7 | composite at 5 1/s [Pa s] | 55 | 54.95 | OK |
| fig7 | linear PTT: eta_P1 at 1 1/s | 35 | 34.923 | OK |
| fig7 | linear PTT: eta_P1 at 5 1/s | 42 | 41.542 | OK |
| fig7 | exponential PTT: eta_P1 at 1 1/s | 24 | 24.301 | OK |
| fig7 | exponential PTT: eta_P1 at 5 1/s | 26 | 25.962 | OK |
| fig7 | Giesekus: eta_P1 at 1 1/s | 35 | 35.239 | OK |
| fig7 | Giesekus: eta_P1 at 5 1/s | 56 | 55.518 | OK |
| fig7 | multimode fits: smallest rms log misfit | 0.14 | 0.14312 | OK |
| fig7 | multimode fits: largest rms log misfit | 0.21 | 0.2065 | OK |
| fig7 | single-mode Rolie-Poly: rms log misfit | 0.21 | 0.21124 | OK |
| fig7 | single mode vs SAOS: rms factor | 3.5 | 3.459 | OK |
| fig7 | composite within 18% of the single-mode Rolie-Poly eta_P1 | <= 0.18 | 0.17489 | OK |
