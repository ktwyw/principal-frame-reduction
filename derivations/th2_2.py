"""Further theory: (1) tests of the necessary-and-sufficient condition (terms that are quadratic in D but purely
off-diagonal in the eigenframe; modified spin tensors); (2) realizability: any off-diagonal rate can be imposed at a
given principal state; (3) short-time error of the rate-controlled protocol for a Maxwell spectrum (symbolic);
(4) O(eps^4) correction for separable K-BKZ fluids with arbitrary memory function (symbolic)."""
import numpy as np, sympy as sp, json
from scipy.optimize import fsolve
from lib import *
out = {}
def dev(m, rates, T=400, flows=('shear', 'mixed')):
    w = 0
    for g in rates:
        for L in ([Lshear(g)] if flows == ('shear',) else [Lshear(g), Lmixed(g, 0.3), Lmixed(g, -0.3)]):
            s = m['stress'](steady(m, L, T=T)); d1, P1, P2 = planar_projection(s, L); x = m['stress'](steady(m, Lext(d1), T=20*T))
            w = max(w, abs(P1/((x[0, 0]-x[1, 1])/d1) - 1), abs(P2/((x[2, 2]-x[1, 1])/d1) - 1))
    return w
print('\n=== 2. realizability: at fixed principal state (a, d) on the planar-extension curve, impose arbitrary off-diagonal rate h ===')
m = giesekus(0.2); e = 0.8; a = np.diag(steady(m, Lext(e), T=4000))
for h in [0.3, 1.0, 3.0]:
    # unknown: spin w (W_12 = w).  Off-diagonal equation in the eigenframe: w(a2 - a1) + h(a1 + a2) - alpha... solved numerically
    def offdiag(w): w = float(np.ravel(w)[0]); L = np.array([[e, h + w, 0], [h - w, -e, 0], [0, 0, 0.]]); return m['rhs'](np.diag(a), L)[0, 1]
    w = fsolve(offdiag, h)[0]; L = np.array([[e, h + w, 0], [h - w, -e, 0], [0, 0, 0.]]); res = np.max(abs(m['rhs'](np.diag(a), L)))
    A = steady(m, L, T=400); print(f'  h = {h}: spin w = {w:.4f}; residual of full steady equations at diag(a) = {res:.1e}; state reached from rest has eigenvalues {np.round(np.linalg.eigvalsh(A), 6)} vs a = {np.round(np.sort(a), 6)}')

json.dump(out, open('theory2_2.json','w'), default=float)
