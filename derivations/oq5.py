"""Stress test of the lower bound and of monotonic shear rate: wide spectra (up to 10 modes over 4 decades), rates from 0.02 to 5 /tau_max."""
import numpy as np
exec(open('oq2.py').read().split("print('(a) proof ingredients')")[0])
rng = np.random.default_rng(77); mx = -9; gmin = 9; emax = -9
for trial in range(60):
    n = rng.integers(2, 11); tk = 10**rng.uniform(-4, 0, n); tk /= tk.max(); hk = rng.dirichlet(np.ones(n)*rng.choice([0.3, 1, 3])); Gk = hk/tk; nk = 1/tk
    eps = 10**rng.uniform(-1.7, 0.7); T = min(40.0, 8/eps); tt = np.geomspace(1e-5, T, 3000); Sk, Nk = protocol(Gk, nk, eps, tt, t0=1e-9)
    S, N = Sk.sum(0), Nk.sum(0); D = np.hypot(N, 2*S); gd = eps*D/S; Dx = planar(Gk, nk, eps, tt); gam = np.concatenate([[0], np.cumsum((gd[1:] + gd[:-1])/2*np.diff(tt))]) + 2*eps*tt[0]
    mx = max(mx, (D/Dx).max() - 1); gmin = min(gmin, (np.diff(gd)/gd[:-1]).min()); emax = max(emax, (gam/(2*np.sinh(eps*tt))).max() - 1)
print('max ratio - 1:', mx, ' min relative increment of shear rate:', gmin, ' max gamma/(2 sinh) - 1:', emax)
