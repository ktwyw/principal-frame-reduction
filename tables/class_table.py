"""Table of models inside / outside the class, with the measured maximum deviation of the composites."""
import numpy as np, json
from lib import *
def dev(m, rates, T=400):
    w = 0
    for g in rates:
        s = m['stress'](steady(m, Lshear(g), T=T)); e, P1, P2 = composite(s, g); x = m['stress'](steady(m, Lext(e), T=20*T))
        w = max(w, abs(P1/((x[0, 0]-x[1, 1])/e) - 1), abs(P2/((x[2, 2]-x[1, 1])/e) - 1))
    return w
R8 = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]; r1 = [0.1, 0.3, 1.0, 3.0]
def wm_rhs():
    def rhs(A, L):
        D = (L + L.T)/2; W = (L - L.T)/2; t = 1/(1 + np.sqrt(2*np.sum(D*D))); return W@A - A@W + D@A + A@D - A/t + 2*D
    return dict(rhs=rhs, stress=lambda A: A, A0=np.zeros((3, 3)))
E = json.load(open('extend.json'))   # written by extend.py
inn = [('UCM ($\\xi=1$)', dev(ucm(), R8, T=2000)), ('Lower-convected Maxwell ($\\xi=-1$)', dev(stress_model(lambda A, D: A, a=-1.0), [0.03, 0.1, 0.3, 1, 3, 10])),
       ('Johnson-Segalman ($\\xi=0.6$)', dev(stress_model(lambda A, D: A, a=0.6), [0.03, 0.1, 0.3, 1])),
       ('PTT, linear kernel ($\\varepsilon=0.1$)', dev(lptt(), R8)), ('PTT, exponential kernel ($\\varepsilon=0.1$)', dev(eptt(), R8)),
       ('Giesekus ($\\alpha_{\\rm G}=0.2$)', dev(giesekus(0.2), R8)), ('FENE-P ($b=50$)', dev(fenep(50.0), R8)), ('Rolie-Poly, finite extensibility', dev(rolie_poly(), R8, T=2000)),
       ('Stress-power-dependent relaxation, $\\phi_1(J_1)$', dev(stress_model(lambda A, D: (1 + 0.3*np.sum(A*D))*A), [0.03, 0.1, 0.3, 1, 3, 10])),
       ('Terms $(\\tr\\bm\\sigma)\\bm D$, $(\\bm\\sigma\\!:\\!\\bm D)\\bm I$; $\\xi=0.8$', E['members']['Oldroyd-type: xi=0.8, +0.1 tr(s) D + 0.1 (s:D) I   [in]']),
       ('Terms $\\bm\\sigma\\!\\cdot\\!\\bm D+\\bm D\\!\\cdot\\!\\bm\\sigma$, $\\bm\\sigma^2\\!\\cdot\\!\\bm D+\\bm D\\!\\cdot\\!\\bm\\sigma^2$', E['members']['s.D + D.s and s^2.D + D.s^2 terms (psi_1, psi_2)     [in]']),
       ('Strand-extension term $(\\bm\\sigma\\!:\\!\\bm D)(\\bm\\sigma+G\\bm I)$', E['members']['Larson partially extending: + (2 xi/3)(D:s) s        [in]']),
       ('Differential pom-pom (tensor + scalar stretch)', max(max(abs(r[4]/r[5]-1), abs(r[6]/r[7]-1)) for r in E['pompom']))]
out = [('Term $\\bm D^2$ (coefficient $0.3\\tau$)', E['members']['D.D term (Reiner-Rivlin-like), kappa = 0.3          [out]']),
       ('Term $\\bm\\sigma\\!\\cdot\\!\\bm D^2+\\bm D^2\\!\\cdot\\!\\bm\\sigma$ ($0.3\\tau^2$)', E['members']['s.D^2 + D^2.s term, kappa = 0.3                     [out]']),
       ('Relaxation rate depending on $\\tr(\\bm\\sigma\\!\\cdot\\!\\bm D^2)$', E['members']['coefficient depending on tr(s D^2)                  [out]']),
       ('White-Metzner, $\\tau(\\tr\\bm D^2)$, constant modulus', dev(wm_rhs(), r1))]
def fmt(v): 
    if v < 1e-9:
        e = int(np.floor(np.log10(max(v, 1e-17)))); return f"${max(v,1e-17)/10**e:.0f}\\times10^{{{e}}}$"
    return f"{100*v:.0f}\\%"
rows = ["\\multicolumn{2}{l}{\\emph{Inside the class}}\\\\"] + [f"{n} & {fmt(v)}\\\\" for n, v in inn] + ["\\hline", "\\multicolumn{2}{l}{\\emph{Outside the class}}\\\\"] + [f"{n} & {fmt(v)}\\\\" for n, v in out] + [
        "Newtonian solvent added (Oldroyd-B) & Fig.~\\ref{fig:outside}\\\\", "Several modes (Maxwell, PTT; Rouse, Zimm) & Figs.~\\ref{fig:spectra}, \\ref{fig:outside}\\\\",
        "K-BKZ integral model & Fig.~\\ref{fig:outside}\\\\", "FENE dumbbells without closure & Table~\\ref{tab:bd}\\\\"]
open('output/class_table.tex', 'w').write(r"""\begin{table}
\caption{Models inside and outside the class of Eqs.~\eqref{class} and \eqref{F}. The second column is the largest relative difference between the composites $\eta_{[P1]}$, $\eta_{[P2]}$ computed from full-tensor steady shear solutions and the planar extensional viscosities computed independently, over the shear rates tested ($0.03\le\gd\tau\le100$ for the named models; $\gd\tau\le3$ for the others, and wherever steady states exist). Unless stated, added terms have coefficients of $0.02$--$0.3$ in units of $\eta_0$ and $\tau$ on a UCM base.}\label{tab:class}
\begin{ruledtabular}
\begin{tabular}{lc}
Model or constitutive term & Max. difference\\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
""")
for n, v in inn + out: print(f'{v:10.2e}  {n}')
