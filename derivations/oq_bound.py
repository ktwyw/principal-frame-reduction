"""Protocol for Maxwell spectra: exact identity for Delta_shear - Delta_planar, kernel sign, w >= 0 and monotone shear rate (random spectra)."""
import numpy as np, json
from scipy.integrate import solve_ivp
def run(Gk, nk, eps, T, npts=4000):
    n = len(Gk); t0 = 1e-7
    def f(tt, y):
        Sk, Nk = y[:n], y[n:]; S_, N_ = Sk.sum(), Nk.sum(); g = eps*np.sqrt(N_**2 + 4*S_**2)/S_
        return np.r_[Gk*g - nk*Sk, 2*g*Sk - nk*Nk]
    tt = np.linspace(t0, T, npts)
    r = solve_ivp(f, (t0, T), np.r_[Gk*2*eps*t0, Gk*4*eps**2*t0**2], t_eval=tt, rtol=1e-11, atol=1e-15, method='LSODA')
    Sk, Nk = r.y[:n], r.y[n:]; S, N = Sk.sum(0), Nk.sum(0); D = np.hypot(N, 2*S); g = eps*D/S
    Sd = (Gk[:, None]*g - nk[:, None]*Sk).sum(0); Nd = (2*g*Sk - nk[:, None]*Nk).sum(0)
    chid = (Sd*N - S*Nd)/D**2                      # d chi/dt, tan 2chi = 2S/N
    w = g/2 + chid; h = g/2*N/D
    Yk = (-Nk*2*S + 2*Sk*N)/D                      # 2 c'_12 of mode k in the total-stress frame (stress units)
    Dx = sum(Gk[k]*2*eps*(-np.expm1(-(nk[k] - 2*eps)*tt)/(nk[k] - 2*eps) - np.expm1(-(nk[k] + 2*eps)*tt)/(nk[k] + 2*eps)) for k in range(n))
    return tt, D, Dx, g, w, h, Yk
# identity check on the two-mode fluid
Gk = np.array([2.5, 0.5/1.8]); nk = np.array([5, 1/1.8]); eps = 0.2
tt, D, Dx, g, w, h, Yk = run(Gk, nk, eps, 10.0, 8001)
for te in (1.0, 3.0, 10.0):
    i = np.searchsorted(tt, te); s = tt[:i+1]; tau = tt[i] - s
    ker = (np.exp(-nk[:, None]*tau)*Yk[:, :i+1]).sum(0); integ = 2*np.trapezoid((w[:i+1]*np.cosh(2*eps*tau) + h[:i+1]*np.sinh(2*eps*tau))*ker, s)
    print(f't={tt[i]:.2f}: Delta_sh - Delta_pl = {D[i]-Dx[i]:.8f}; identity integral = {integ:.8f}')
rng = np.random.default_rng(5); res = []
for trial in range(200):
    n = rng.integers(2, 6); tk = 10**rng.uniform(-2, 0, n); hk = rng.dirichlet(np.ones(n)); Gk = hk/tk; nk = 1/tk
    eps = rng.uniform(0.05, 1.5)*0.5/tk.max() if trial % 2 else rng.uniform(0.05, 0.48)/tk.max()   # includes rates above 1/2tau_max
    T = min(40.0, 8/eps); tt, D, Dx, g, w, h, Yk = run(Gk, nk, eps, T, 3000)
    # kernel sign at a few lags
    ks = max(((np.exp(-nk[:, None]*lag)*Yk).sum(0)).max()/np.abs(Yk).max() for lag in (0.01, 0.1, 1.0))
    res.append([(D/Dx).max() - 1, w.min()/g.max(), np.diff(g).min()/g.max(), ks, (D/Dx).min()])
res = np.array(res)
print('200 random spectra (2-5 modes, two decades, rates up to 0.75/tau_max):')
print('  max(ratio) - 1            :', res[:, 0].max()); print('  min w / max gdot          :', res[:, 1].min())
print('  min increment of gdot     :', res[:, 2].min()); print('  max of kernel sum (scaled):', res[:, 3].max()); print('  min ratio:', res[:, 4].min())
json.dump(dict(res=res.tolist()), open('oq_bound.json', 'w'))
