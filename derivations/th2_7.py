"""Protocol for random Maxwell spectra: is Delta_shear/Delta_ext monotone decreasing from 1 to the steady ratio?"""
import numpy as np, json
from scipy.integrate import solve_ivp
rng = np.random.default_rng(3); res = []
for trial in range(60):
    n = rng.integers(2, 5); tk = 10**rng.uniform(-1.5, 0, n); tk /= tk.max(); hk = rng.dirichlet(np.ones(n)); Gk = hk/tk; nk = 1/tk
    eps = rng.uniform(0.05, 0.45); t0 = 1e-7; T = 60.0
    def f(tt, y):
        Sk, Nk = y[:n], y[n:]; S_, N_ = Sk.sum(), Nk.sum(); g = eps*np.sqrt(N_**2 + 4*S_**2)/S_
        return np.r_[Gk*g - nk*Sk, 2*g*Sk - nk*Nk]
    tt = np.geomspace(1e-3, T, 600); r = solve_ivp(f, (t0, T), np.r_[Gk*2*eps*t0, Gk*4*eps**2*t0**2], t_eval=tt, rtol=1e-11, atol=1e-16, method='LSODA')
    S_, N_ = r.y[:n].sum(0), r.y[n:].sum(0); Dsh = np.sqrt(N_**2 + 4*S_**2)
    Dx = sum(Gk[k]*2*eps*(-np.expm1(-(nk[k] - 2*eps)*tt)/(nk[k] - 2*eps) - np.expm1(-(nk[k] + 2*eps)*tt)/(nk[k] + 2*eps)) for k in range(n))
    ra = Dsh/Dx; tb = (hk*tk).sum(); st = (1/(1 - 4*tb**2*eps**2))/(hk/(1 - 4*tk**2*eps**2)).sum()
    res.append((ra.max() - 1, np.diff(ra).max(), ra[-1], st))
res = np.array(res); print('max(ratio)-1:', res[:, 0].max(), ' max increment:', res[:, 1].max(), ' max |final - steady ratio|:', abs(res[:, 2] - res[:, 3]).max(), ' range of steady ratio', res[:, 3].min(), res[:, 3].max())
