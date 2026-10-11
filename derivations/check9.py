"""v9 re-checks with explicit definitions: (a) multimode linear-PTT scan; (b) start-up shear vs start-up planar extension for linear PTT."""
import numpy as np, json
from scipy.integrate import solve_ivp
from lib import *
out = {}
# (a) 600 random spectra per value of eps: 2-4 modes, tau log-uniform over two decades, weights Dirichlet(1), common eps
for ep in (0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
    rng = np.random.default_rng(11); lo, hi = 9, 0
    for trial in range(600):
        n = rng.integers(2, 5); tk = 10**rng.uniform(-2, 0, n); hk = rng.dirichlet(np.ones(n)); tb = (hk*tk).sum(); modes = [(h, t/tb, ep) for h, t in zip(hk, tk)]
        for g in np.logspace(-1.5, 4, 45):
            x, c, tr = lptt_multi(modes, g); lo, hi = min(lo, c/tr), max(hi, c/tr)
    out[f'scan_{ep}'] = (lo, hi); print(f'eps = {ep}: ratio composite/planar in [{lo:.3f}, {hi:.3f}]', flush=True)
# (b) linear PTT eps = 0.1, gdot*tau = 5
m = lptt(0.1); g = 5.0; f = lambda t, v: m['rhs'](sym(v), Lshear(g))[IU]
T = 12; tt = np.linspace(1e-3, T, 4000); s = solve_ivp(f, [0, T], np.zeros(6), t_eval=tt, method='Radau', rtol=1e-10, atol=1e-13)
S, N1 = s.y[1], s.y[0] - s.y[3]; Dl = np.hypot(N1, 2*S); ee = g*S/Dl; einf = ee[-1]
fe = lambda t, v: np.diag(m['rhs'](np.diag(v), Lext(einf))); x = solve_ivp(fe, [0, T], np.zeros(3), t_eval=tt, method='Radau', rtol=1e-10, atol=1e-13); Dx = x.y[0] - x.y[1]
H = np.concatenate([[0], np.cumsum((ee[1:] + ee[:-1])/2*np.diff(tt))]) + g/2*tt[0]
print(f'terminal eps_eff tau = {einf:.4f}; eps_eff(0+) tau = {g/2}')
r1 = (Dl/ee)/(Dx/einf); print(f'(i) transient composite Dl(t)/eps_eff(t) vs planar eta+(t) at the same time: ratio min {r1.min():.3f} max {r1.max():.3f}; planar/composite - 1 max {(1/r1).max()-1:.3f}')
r2 = (Dl/ee)/(np.interp(H, einf*tt, Dx)/einf); print(f'(ii) same, compared at equal Hencky strain (int eps_eff dt vs einf t): ratio min {r2.min():.3f} max {r2.max():.3f}')
r3 = Dl/np.interp(H, einf*tt, Dx); print(f'(iii) stress difference Dl at equal Hencky strain: ratio min {r3[5:].min():.3f} max {r3.max():.3f}')
r4 = Dl/Dx; print(f'(iv) stress difference at equal time: max {r4.max():.3f}')
out['startup'] = dict(einf=einf, r1=(r1.min(), r1.max()), r2=(r2.min(), r2.max()), r3=(r3[5:].min(), r3.max()), r4=r4.max())
json.dump(out, open('check9.json', 'w'))
