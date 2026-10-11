import numpy as np
exec(open('oq2.py').read().split("print('(a) proof ingredients')")[0])
rng = np.random.default_rng(21); worst = 9; nneg = 9; cnt = 0
for trial in range(40):
    n = rng.integers(2, 6); tk = np.sort(10**rng.uniform(-2, 0, n)); tk /= tk.max(); hk = rng.dirichlet(np.ones(n)); Gk = hk/tk; nk = 1/tk
    eps = rng.uniform(0.05, 1.5); T = min(30.0, 5/eps); tt = np.linspace(1e-4, T, 4001); Sk, Nk = protocol(Gk, nk, eps, tt)
    S, N = Sk.sum(0), Nk.sum(0); D = np.hypot(N, 2*S); gd = eps*D/S; sig = Gk[:, None]*gd - nk[:, None]*Sk; nn = 2*gd*Sk - nk[:, None]*Nk
    ok = (nn/Nk > 1e-6).all(0) & (np.arange(len(tt)) > 20); order = np.argsort(nk); r = (sig/nn)[order][:, ok]
    if ok.sum(): worst = min(worst, (np.diff(r, axis=0)/r[:-1]).min()); cnt += ok.sum()
    nneg = min(nneg, (nn/Nk)[:, 20:].min())
print('min relative increment of sigma_k/n_k with nu_k:', worst, ' samples', cnt, ' min n_k/N_k', nneg)
