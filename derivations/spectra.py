"""Rouse and Zimm chains (multimode UCM, tau_p = tau_1 p^-k, equal moduli): composite vs true planar viscosity."""
import numpy as np
from scipy.special import zeta
def chain(k, w, P=200000):
    p = np.arange(1, P+1.0); t = p**-k; eta0 = t.sum(); tb = (t*t).sum()/eta0
    true = 4*(t/(1 - 4*t*t*w*w)).sum()/eta0; comp = 4/(1 - 4*tb*tb*w*w) if 2*tb*w < 1 else np.inf
    return true, comp
for name, k in [('Rouse', 2.0), ('Zimm', 1.5)]:
    tb = zeta(2*k)/zeta(k); t2 = zeta(3*k)/zeta(k)
    print(f'{name}: tbar/tau1 = {tb:.6f}; <tau^2>/tbar^2 = {t2/tb**2:.6f}; fraction of leading rise captured = {tb**2/t2:.4f}; '
          f'Var/tau1^2 = {t2-tb**2:.6f}; composite pole at eps*tau1 = {1/(2*tb):.4f} (true 0.5)')
    for w in [0.1, 0.2, 0.3, 0.4, 0.45]:
        tr, c = chain(k, w); print(f'   eps*tau1={w}: true eta_P1/eta0 = {tr:.4f}, composite = {c:.4f}, ratio = {c/tr:.4f}')
print('exact Rouse: tbar = pi^2/15 =', np.pi**2/15, '; <tau^2>/tbar^2 = 10/7 =', 10/7)
# two-mode pair used in the audit
p1 = lambda e, modes: sum(4*h/(1-4*t*t*e*e) for h, t in modes)
print('two-mode pair at eps=0.25:', p1(.25, [(1, 1)]), p1(.25, [(.5, .2), (.5, 1.8)]))
