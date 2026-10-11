"""Other shear-free flows from the principal-frame reduction: simple shear with a superposed stretch/compression
along the vorticity axis, L = gdot e1e2 + diag(-a/2, -a/2, a), maps to uniaxial (a<0) or equibiaxial (a>0) extension
when a is tuned so that two projected rates coincide."""
import numpy as np, json
from scipy.optimize import brentq
from lib import *
def proj(m, g, a):
    L = np.array([[-a/2, g, 0], [0, -a/2, 0], [0, 0, a]]); A = steady(m, L, T=300); s = m['stress'](A); D = (L + L.T)/2
    w, V = np.linalg.eigh(s[:2, :2]); n = [np.array([V[0, k], V[1, k], 0.0]) for k in (1, 0)]
    return (n[0]@D@n[0], n[1]@D@n[1], a), (w[1], w[0], s[2, 2])          # (d+, d-, d3), (s+, s-, s3)
def ext3(m, d):
    s = m['stress'](steady(m, np.diag(d), T=3000)); return np.diag(s)
out = {}
for name, m in [('linear PTT', lptt()), ('Giesekus', giesekus(0.2)), ('FENE-P', fenep(50.0)), ('Rolie-Poly', rolie_poly())]:
    rows = []
    for g in [0.3, 1.0, 3.0, 10.0]:
        # uniaxial: need d- = d3 (a < 0)
        a = brentq(lambda a: proj(m, g, a)[0][1] - a, -0.66*g/2, -1e-6*g, xtol=1e-13)
        d, s = proj(m, g, a); eps = d[0]; etaE_c = (s[0] - s[1])/eps; su = ext3(m, [eps, -eps/2, -eps/2]); etaE = (su[0] - su[1])/eps
        # equibiaxial: need d+ = d3 (a > 0)
        try:
            b = brentq(lambda b: proj(m, g, b)[0][0] - b, 1e-6*g, 0.5*g, xtol=1e-13); d2, s2 = proj(m, g, b); eb = d2[0]
            etaB_c = (s2[0] - s2[1])/eb; sb = ext3(m, [eb, -2*eb, eb]); etaB = (sb[0] - sb[1])/eb
        except Exception as ex: b = eb = etaB_c = etaB = np.nan
        rows.append([g, a, eps, etaE_c, etaE, abs(s[1] - s[2])/abs(s[0]), b, eb, etaB_c, etaB])
        print(f'{name:11s} gdot={g:5.1f}: a/gdot={a/g:+.4f} -> uniaxial rate {eps:.4f}: eta_E mapped {etaE_c:.6f} vs true {etaE:.6f} (rel {abs(etaE_c/etaE-1):.1e}; |s- - s3|/s+ = {rows[-1][5]:.1e})'
              f' | a/gdot={b/g:+.4f} -> biaxial rate {eb:.4f}: mapped {etaB_c:.6f} vs true {etaB:.6f} (rel {abs(etaB_c/etaB-1):.1e})')
    out[name] = rows
json.dump(out, open('other_ext.json', 'w'), default=float)
