"""Checks: (a) protocol series to higher order + numerical confirmation; (b) multimode K-BKZ O(eps^4) vs quadrature."""
import sympy as sp, numpy as np, mpmath as mp
from scipy.integrate import solve_ivp
src = open('th2_3.py').read().replace('NORD = 5', 'NORD = 7').replace("t, 0, 5)", "t, 0, 7)")
src = src[:src.index('# numerics')]
exec(src)
print('  ratio to t^6:', ratio)
modes = [(.5, .2), (.5, 1.8)]; mm = [sum(h/tk*(1/tk)**j for h, tk in modes) for j in range(8)]
Gk = np.array([h/tk for h, tk in modes]); nk = np.array([1/tk for h, tk in modes])
for eps in (0.25, 1.0):
    pr = sp.lambdify(t, ratio.subs({ms[j]: mm[j] for j in range(8)}).subs(e0, eps)); t0 = 1e-8
    def f(tt, y):
        Sk, Nk = y[:2], y[2:]; S_, N_ = Sk.sum(), Nk.sum(); gdt = eps*np.sqrt(N_**2 + 4*S_**2)/S_
        return np.r_[Gk*gdt - nk*Sk, 2*gdt*Sk - nk*Nk]
    tt = np.array([0.05, 0.1, 0.2, 0.4]); r = solve_ivp(f, (t0, tt[-1]), np.r_[Gk*2*eps*t0, Gk*4*eps**2*t0**2], t_eval=tt, rtol=1e-13, atol=1e-18, method='DOP853')
    S_, N_ = r.y[:2].sum(0), r.y[2:].sum(0); Dsh = np.sqrt(N_**2 + 4*S_**2)
    Dx = sum(Gk[k]*2*eps*(-np.expm1(-(nk[k] - 2*eps)*tt)/(nk[k] - 2*eps) - np.expm1(-(nk[k] + 2*eps)*tt)/(nk[k] + 2*eps)) for k in range(2))
    var = mm[2]/mm[0] - (mm[1]/mm[0])**2
    print(f'  eps0={eps}: (1 - ratio)/(eps^2 Var t^4/120) numeric:', np.round((1 - Dsh/Dx)/(eps**2*var*tt**4/120), 4), ' series:', np.round([(1 - pr(q))/(eps**2*var*q**4/120) for q in tt], 4))
# (b) K-BKZ multimode
mp.mp.dps = 30
def kbkz(modes, a, hfun):
    mem = lambda s: sum(h/tk**2*mp.e**(-s/tk) for h, tk in modes)
    def shear(g):
        S = mp.quad(lambda s: mem(s)*hfun(g*g*s*s)*g*s, [0, 1, 10, 100, mp.inf]); N = mp.quad(lambda s: mem(s)*hfun(g*g*s*s)*g*g*s*s, [0, 1, 10, 100, mp.inf]); return S, N
    def planar(e): return mp.quad(lambda s: mem(s)*hfun(4*mp.sinh(e*s)**2)*2*mp.sinh(2*e*s), [0, 1, 10, 100, 400])/e
    return shear, planar
for hname, hfun_ in (('PSM 1/(1+aI)', lambda a: (lambda I: 1/(1 + a*I))), ('exp(-aI)... Wagner-type', lambda a: (lambda I: mp.e**(-a*I)))):
    for modes, a in (([(.5, .2), (.5, 1.8)], 0.05), ([(.3, .5), (.7, 1.3)], 0.2), ([(1.0, 1.0)], 0.1)):
        shear, planar = kbkz(modes, a, hfun_(a)); mu = [sum(h*tk**n for h, tk in modes) for n in range(5)]
        c2 = 16*(mu[2] - mu[1]**2); c4 = 64*(mu[4] - mu[1]**4) - 384*a*(5*mu[4] - 4*mu[1]*mu[3])
        res = []
        for e in (0.01, 0.02):
            g = mp.findroot(lambda g: (lambda SN: g*SN[0]/mp.sqrt(SN[1]**2 + 4*SN[0]**2))(shear(g)) - e, 2*e); S, N = shear(g)
            comp = (N**2 + 4*S**2)/(g*S); res.append(float((planar(e) - comp - c2*e**2)/e**4))
        print(f'  K-BKZ {hname}, modes {modes}, a={a}: predicted eps^4 coeff {c4:.4f}; numerical (eps=0.01, 0.02): {res[0]:.4f}, {res[1]:.4f}')
