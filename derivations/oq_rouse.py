"""Protocol for the Rouse spectrum (tau_p = tau_1/p^2, equal moduli, 200 modes): ratio Delta_shear/Delta_planar versus Hencky strain."""
import numpy as np, json
from scipy.integrate import solve_ivp
p = np.arange(1, 201); tk = 1/p**2.0; Gk = np.ones_like(tk); nk = 1/tk; n = len(tk); out = {}
xs = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
for Wi in (0.2, 0.4, 1.0, 3.0, 10.0, 30.0):
    eps = Wi; t0 = 1e-9; ts = xs/eps
    def f(tt, y):
        Sk, Nk = y[:n], y[n:]; S_, N_ = Sk.sum(), Nk.sum(); g = eps*np.sqrt(N_**2 + 4*S_**2)/S_
        return np.r_[Gk*g - nk*Sk, 2*g*Sk - nk*Nk]
    r = solve_ivp(f, (t0, ts[-1]), np.r_[Gk*2*eps*t0, Gk*4*eps**2*t0**2], t_eval=ts, rtol=1e-9, atol=1e-30, method='LSODA')
    S_, N_ = r.y[:n].sum(0), r.y[n:].sum(0); D = np.hypot(N_, 2*S_)
    Dx = sum(Gk[k]*2*eps*(-np.expm1(-(nk[k] - 2*eps)*ts)/(nk[k] - 2*eps) - np.expm1(-(nk[k] + 2*eps)*ts)/(nk[k] + 2*eps)) for k in range(n))
    tb = (tk**2).sum()/tk.sum(); st = (1/(1 - 4*tb**2*eps**2))/((tk/(1 - 4*tk**2*eps**2)).sum()/tk.sum()) if eps < 0.5 else float('nan')
    out[Wi] = (D/Dx).tolist(); print(f'eps0 tau1 = {Wi:5.1f}: ratio at Hencky strain {xs.tolist()} =', np.round(D/Dx, 3).tolist(), f'| steady ratio {st:.3f}')
json.dump(out, open('oq_rouse.json', 'w'))
