"""Universality near shear-free flows: mismatch of the principal-frame reduction is second order in the departure from planar extension,
for fluids outside the class.  Mixed flows L = g (e1e2 + alpha e2e1), alpha = 1 - delta."""
import numpy as np, json
from lib import *
out = {}
def ucm_modes(modes):
    def stress(L):
        return sum(h/tk*steady(ucm(), L*tk, T=200) for h, tk in modes)
    def planar(e): return sum(4*h/(1 - 4*(tk*e)**2) for h, tk in modes)
    return stress, planar
modes = [(.5, .2), (.5, 1.8)]; stress, planar = ucm_modes(modes)
print('two-mode Maxwell fluid, base rate 0.2:')
for dl in (0.4, 0.2, 0.1, 0.05, 0.025):
    al = 1 - dl; gdot = 0.2*2/(1 + al); L = Lmixed(gdot, al); s = stress(L); d1, P1, P2 = planar_projection(s, L); mis = P1/planar(d1) - 1
    out[f'maxwell_{dl}'] = mis; print(f'  delta = {dl:6.3f}: d1 = {d1:.5f}; composite/planar - 1 = {mis:.3e}; /delta^2 = {mis/dl**2:.4f}')
print('single tensor outside the class (UCM + 0.3 (A.D^2 + D^2.A)), base rate 0.3:')
m = stress_model(lambda A, D: A + 0.3*(A@D@D + D@D@A))
for dl in (0.4, 0.2, 0.1, 0.05, 0.025):
    al = 1 - dl; gdot = 0.3*2/(1 + al); L = Lmixed(gdot, al); s = steady(m, L); d1, P1, P2 = planar_projection(s, L); x = steady(m, Lext(d1)); mis = P1/((x[0, 0] - x[1, 1])/d1) - 1
    out[f'AD2_{dl}'] = mis; print(f'  delta = {dl:6.3f}: composite/planar - 1 = {mis:.3e}; /delta^2 = {mis/dl**2:.4f}')
json.dump(out, open('oq_nearsf.json', 'w'))
