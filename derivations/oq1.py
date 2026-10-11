"""Protocol on Maxwell spectra: test the ingredients of a proof of the lower bound.
Rotating principal frame: e = eps0, h = (gd/2)cos2chi, wt = gd/2 + chidot.  Identity:
  Dx(t) - D(t) = -2 int_0^t [wt(s) cosh(2 e (t-s)) + h(s) sinh(2 e (t-s))] B(t,s) ds,  B(t,s) = sum_k exp(-nu_k (t-s)) y_k(s)."""
import numpy as np
from scipy.integrate import solve_ivp
rng = np.random.default_rng(5); worst = dict(wt=9, wtmh=9, gdp=9, Sp=9, B=-9, ident=0, ratio=0)
for trial in range(80):
    n = rng.integers(2, 6); tk = 10**rng.uniform(-2, 0, n); tk /= tk.max(); hk = rng.dirichlet(np.ones(n)); Gk = hk/tk; nk = 1/tk
    eps = rng.uniform(0.05, 1.5) if trial % 2 else rng.uniform(0.05, 0.45); t0 = 1e-7; T = min(40.0, 6/eps)
    def gd_of(y):
        S_, N_ = y[:n].sum(), y[n:].sum(); return eps*np.sqrt(N_**2 + 4*S_**2)/S_
    f = lambda tt, y: np.r_[Gk*gd_of(y) - nk*y[:n], 2*gd_of(y)*y[:n] - nk*y[n:]]
    tt = np.linspace(1e-3, T, 3000); r = solve_ivp(f, (t0, T), np.r_[Gk*2*eps*t0, Gk*4*eps**2*t0**2], t_eval=tt, rtol=1e-11, atol=1e-15, method='LSODA')
    Sk, Nk = r.y[:n], r.y[n:]; S, N = Sk.sum(0), Nk.sum(0); D = np.hypot(N, 2*S); gd = eps*D/S
    Sp = Gk.sum()*gd - (nk[:, None]*Sk).sum(0); Np = 2*gd*S - (nk[:, None]*Nk).sum(0); chid = (Sp*N - S*Np)/D**2
    wt = gd/2 + chid; h = gd/2*N/D; y = (2*Sk*N - 2*S*Nk)/D          # y_k in the total principal frame
    R = N/S; gdp = eps*R*(Np*S - N*Sp)/S**2/np.sqrt(R**2 + 4)
    Dx = sum(Gk[k]*2*eps*(-np.expm1(-(nk[k] - 2*eps)*tt)/(nk[k] - 2*eps) - np.expm1(-(nk[k] + 2*eps)*tt)/(nk[k] + 2*eps)) for k in range(n))
    # identity check at the final time
    Bts = (np.exp(-nk[:, None]*(T - tt))*y).sum(0); K = wt*np.cosh(2*eps*(T - tt)) + h*np.sinh(2*eps*(T - tt))
    rhs = -2*np.trapezoid(K*Bts, tt); lhs = Dx[-1] - D[-1]
    # B(t,s) <= 0 for several t
    Bmax = max(((np.exp(-nk[:, None]*(tt[i] - tt[:i+1]))*y[:, :i+1]).sum(0)).max()/D[i] for i in range(100, 3000, 300))
    for k_, v in (('wt', (wt/gd).min()), ('wtmh', ((wt - h)/gd).min()), ('gdp', (gdp/gd).min()), ('Sp', (Sp/S).min())): worst[k_] = min(worst[k_], v)
    worst['B'] = max(worst['B'], Bmax); worst['ident'] = max(worst['ident'], abs(rhs/lhs - 1) if abs(lhs) > 1e-6*Dx[-1] else 0); worst['ratio'] = max(worst['ratio'], (D/Dx).max() - 1)
print(worst)
