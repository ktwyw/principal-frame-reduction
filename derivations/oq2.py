"""Open questions, part 2.
(a) proof ingredients for the protocol lower bound, checked to high accuracy;
(b) power-law relaxation G ~ t^-alpha: universal error function of Hencky strain, leading coefficient kappa(alpha);
(c) general O(eps^4) composite coefficient in terms of shear expansion coefficients; consistency with K-BKZ;
(d) dumbbell Fokker-Planck: residual of the generalized Boltzmann ansatz."""
import numpy as np, sympy as sp, json
from scipy.integrate import solve_ivp, simpson
out = {}
def protocol(Gk, nk, eps, tt, t0=1e-6):
    n = len(Gk); g0 = 2*eps
    def gd_of(y):
        S_, N_ = y[:n].sum(), y[n:].sum(); return eps*np.sqrt(N_**2 + 4*S_**2)/S_
    f = lambda t, y: np.r_[Gk*gd_of(y) - nk*y[:n], 2*gd_of(y)*y[:n] - nk*y[n:]]
    x = nk*t0; y0 = np.r_[Gk*g0*(-np.expm1(-x))/nk, 2*Gk*g0**2*np.where(x < 1e-4, x**2/2 - x**3/3, 1 - np.exp(-x)*(1 + x))/nk**2]
    r = solve_ivp(f, (t0, tt[-1]), y0, t_eval=tt, rtol=1e-11, atol=1e-16, method='LSODA'); return r.y[:n], r.y[n:]
def planar(Gk, nk, eps, tt): return sum(G*2*eps*(-np.expm1(-(v - 2*eps)*tt)/(v - 2*eps) - np.expm1(-(v + 2*eps)*tt)/(v + 2*eps)) for G, v in zip(Gk, nk))
print('(a) proof ingredients')
rng = np.random.default_rng(8); W = dict(ident=0, B=-9, ord1=9, ord2=9, nmin=9, wt=9, gdp=9, wform=0)
for trial in range(40):
    n = rng.integers(2, 6); tk = np.sort(10**rng.uniform(-2, 0, n)); tk /= tk.max(); hk = rng.dirichlet(np.ones(n)); Gk = hk/tk; nk = 1/tk
    eps = rng.uniform(0.05, 1.5); T = min(30.0, 5/eps); tt = np.linspace(1e-4, T, 20001); Sk, Nk = protocol(Gk, nk, eps, tt)
    S, N = Sk.sum(0), Nk.sum(0); D = np.hypot(N, 2*S); gd = eps*D/S; sig = Gk[:, None]*gd - nk[:, None]*Sk; nn = 2*gd*Sk - nk[:, None]*Nk; Sp, Np = sig.sum(0), nn.sum(0)
    wt = gd/2 + (Sp*N - S*Np)/D**2; h = gd*N/(2*D); y = (2*Sk*N - 2*S*Nk)/D
    W['wform'] = max(W['wform'], np.max(abs(wt*D**2 - (gd*N**2/2 + Sp*N + S*(nk[:, None]*Nk).sum(0)))/(gd*D**2)))
    B = (np.exp(-nk[:, None]*(T - tt))*y).sum(0); K = wt*np.cosh(2*eps*(T - tt)) + h*np.sinh(2*eps*(T - tt)); lhs = planar(Gk, nk, eps, tt)[-1] - D[-1]
    W['ident'] = max(W['ident'], abs(-2*simpson(K*B, x=tt)/lhs - 1)); W['B'] = max(W['B'], (B/D[-1]).max())
    order = np.argsort(nk); i0 = 200                                         # nu increasing
    W['ord1'] = min(W['ord1'], np.diff((Sk/Nk)[order][:, i0:], axis=0).min()); W['ord2'] = min(W['ord2'], np.diff((sig/nn)[order][:, i0:], axis=0).min())
    W['nmin'] = min(W['nmin'], (nn/Nk)[:, i0:].min()); W['wt'] = min(W['wt'], (wt/gd).min()); W['gdp'] = min(W['gdp'], np.diff(gd).min()/gd[0])
print({k: float(v) for k, v in W.items()}); out['proof'] = {k: float(v) for k, v in W.items()}
print('(b) power-law relaxation')
def powerlaw(alpha, eps=1.0, Hs=(0.25, 0.5, 1, 1.5, 2, 3, 4)):
    lv = np.linspace(-6, 6, 241); nk = 10**lv; Gk = nk**alpha                # G(t) ~ t^-alpha for 1e-6 << t << 1e6
    tt = np.array(Hs)/eps; Sk, Nk = protocol(Gk, nk, eps, np.r_[1e-3, tt], t0=1e-9); D = np.hypot(Nk.sum(0), 2*Sk.sum(0))[1:]; gam = None
    return D/planar(Gk, nk, eps, tt)
for al in (0.5, 2/3, 0.25):
    kap = 2*al*(1 - al)/((3 - al)*(2 - al)**3); r = powerlaw(al); r2 = powerlaw(al, eps=0.3)
    print(f'  alpha = {al:.3f}: kappa = {kap:.5f}; ratio at H = 0.25,0.5,1,1.5,2,3,4: {np.round(r, 4)}; (1-ratio)/H^2 at H=0.25: {(1-r[0])/0.0625:.5f}; same curve at eps0=0.3: {np.round(r2, 4)}')
    out[f'pl_{al:.3f}'] = dict(kappa=kap, H=[0.25, 0.5, 1, 1.5, 2, 3, 4], ratio=r.tolist())
print('(c) general O(eps^4)')
b1, b2, s3, s5, n4, g, e = sp.symbols('b1 b2 s3 s5 n4 gamma epsilon'); c3, c5 = sp.symbols('c3 c5')
S_ = b1*g + s3*g**3 + s5*g**5; N_ = -2*b2*g**2 + n4*g**4; Dl = sp.sqrt(N_**2 + 4*S_**2)
ee = sp.series(g*S_/Dl, g, 0, 6).removeO(); gs = 2*e + c3*e**3 + c5*e**5; eq = sp.expand(sp.series(ee.subs(g, gs), e, 0, 6).removeO() - e)
so = sp.solve([eq.coeff(e, 3), eq.coeff(e, 5)], [c3, c5], dict=True)[0]; comp = sp.expand(sp.series((Dl**2/(g*S_)).subs(g, gs.subs(so)), e, 0, 5).removeO())
C2 = sp.factor(comp.coeff(e, 2)); C4 = sp.simplify(comp.coeff(e, 4)); print('  C2 =', C2); print('  C4 =', sp.expand(C4)); out['C4'] = str(sp.expand(C4))
a_, bb = sp.symbols('a b'); M = sp.symbols('M1:8'); sub = {b1: M[0], b2: -M[1]/2, s3: -a_*M[2], s5: bb*M[4], n4: -a_*M[3]}
p4 = sp.Rational(64, 120)*M[4]*1 - 0   # planar eps^4 coefficient for K-BKZ computed directly below
s = sp.symbols('s'); mint = lambda ex: sum(sp.expand(ex).coeff(s, j)*M[j - 1] for j in range(1, 8))
I1 = sp.series(4*sp.sinh(e*s)**2, e, 0, 6).removeO(); tru = sp.expand(mint(sp.series(sp.expand((1 - a_*I1 + bb*I1**2)*sp.series(2*sp.sinh(2*e*s), e, 0, 7).removeO()), e, 0, 6).removeO())/e)
d4 = sp.simplify(tru.coeff(e, 4) - C4.subs(sub)); ref = sp.Rational(4, 15)*(2*M[4] - 15*M[1]**4/M[0]**3 - 60*a_*(M[4] - 2*M[1]*M[3]/M[0]))
print('  K-BKZ consistency (should be 0):', sp.simplify(d4 - ref))
print('(d) dumbbell Fokker-Planck with psi = exp(-U + Q.M.Q)')
Q = sp.Matrix(sp.symbols('q1 q2 q3')); fS = sp.Function('f'); Wi = sp.symbols('Wi', positive=True); kap_ = sp.Matrix([[0, Wi, 0], [0, 0, 0], [0, 0, 0]])
C = sp.Matrix([[1 + 2*Wi**2, Wi, 0], [Wi, 1, 0], [0, 0, 1]]); Mh = (sp.eye(3) - C.inv())/2            # Hookean steady state, lambda_H = 1/2 units: kC + Ck^T - C + I = 0
res = lambda Mm, f: sp.simplify(-Mm.trace() + ((2*Mm - f*sp.eye(3))*Q).dot((kap_ - Mm)*Q))
print('  Hookean, simple shear: residual =', res(Mh, 1), '; trace M =', sp.simplify(Mh.trace()))
print('  same M with FENE spring f = 1/(1 - Q^2/b): residual =', sp.factor(res(Mh, 1/(1 - Q.dot(Q)/sp.Symbol('b')))))
json.dump(out, open('oq2.json', 'w'))
