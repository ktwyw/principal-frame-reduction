"""Protocol for a Lodge fluid with power-law relaxation G = K t^-alpha (Rouse-like, alpha = 1/2): scale-free problem in x = eps0 t."""
import sympy as sp, numpy as np, json
from scipy.integrate import solve_ivp
from scipy.special import gamma as Gam
al, t, s, e0 = sp.symbols('alpha t s epsilon_0', positive=True)
NORD = 5; g = sp.symbols('g0:%d' % NORD)
def conv(n):                                                     # int_0^t (t-s)^-alpha s^n ds / Gamma(1-alpha) = n! t^(n+1-alpha)/[(1-alpha)...(n+1-alpha)]
    return t**(n + 1 - al)*sp.factorial(n)/sp.prod([(j + 1 - al) for j in range(n + 1)])
S = sum(g[n]*conv(n) for n in range(NORD))
gam_t = sum(g[n]*t**(n + 1)/(n + 1) for n in range(NORD))
N = 2*gam_t*S - 2*sum(g[n]*g[m]/(m + 1)*conv(n + m + 1) for n in range(NORD) for m in range(NORD))
trunc = lambda e, k: sum(sp.expand(e).coeff(t, i)*t**i for i in range(k + 1))
Sh = trunc(sp.expand(S/t**(1 - al)), NORD); Nh = trunc(sp.expand(N/t**(2 - al)), NORD)
gd = sum(g[n]*t**n for n in range(NORD))
con = sp.expand(gd**2*Sh**2 - e0**2*(t**2*Nh**2 + 4*Sh**2)); sol = {g[0]: 2*e0}
for i in range(1, NORD):
    eq = sp.factor(con.coeff(t, i).subs(sol)); sol[g[i]] = sp.factor(sp.solve(eq, g[i])[0])
for i in range(NORD): print(f'g{i} =', sol[g[i]], flush=True)
Dext = 4*e0*sum((2*e0)**(2*j)*t**(2*j)/(sp.factorial(2*j)*(2*j + 1 - al)) for j in range(4))      # / (Gamma(1-alpha) K t^(1-alpha)) up to the common factor
Shs = trunc(Sh.subs(sol), 4); Nhs = trunc(Nh.subs(sol), 4)
Q = trunc(sp.expand(t**2*Nhs**2 + 4*Shs**2), 4); q0 = Q.coeff(t, 0); u = sp.expand(Q/q0 - 1)      # Dsh = sqrt(q0) sqrt(1+u)
sq = trunc(sp.expand(1 + u/2 - u**2/8), 4); De = trunc(sp.expand(Dext/Dext.subs(t, 0)), 4) ; v = De - 1
inv = trunc(sp.expand(1 - v + v**2), 4); ratio = trunc(sp.expand(sq*inv), 4)
pref = sp.simplify(sp.sqrt(q0)/Dext.subs(t, 0)); print('prefactor (must be 1):', pref)
c2 = sp.factor(sp.cancel(sp.together(ratio.coeff(t, 2)/e0**2))); print('c2 =', c2, '; c1 =', sp.cancel(ratio.coeff(t, 1)), flush=True)
c4n = {}
for a in (sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 4)):
    c4n[a] = sp.nsimplify(sp.cancel((ratio.coeff(t, 4)/e0**4).subs(al, a))); print(f'  alpha = {a}: c2 = {float(c2.subs(al, a)):.6f}, c4 = {float(c4n[a]):.6f}, c3 = {sp.cancel(ratio.coeff(t, 3).subs(al, a))}; g2/eps0^3 = {float((sol[g[2]]/e0**3).subs(al, a)):.5f}', flush=True)
out = dict(c2=str(c2), g2=str(sol[g[2]]), c4={str(k): float(v) for k, v in c4n.items()})
# numerics: power law represented by a dense discrete spectrum (5 modes per decade over 14 decades), eps0 = 1
def numerics(a, xs):
    lt = np.arange(-9, 5.01, 0.2); tk = 10**lt; Gk = tk**(-a)*0.2*np.log(10)/Gam(a); nk = 1/tk; n = len(tk); eps = 1.0; t0 = 1e-12
    def f(tt, y):
        Sk, Nk = y[:n], y[n:]; S_, N_ = Sk.sum(), Nk.sum(); gg = eps*np.sqrt(N_**2 + 4*S_**2)/S_
        return np.r_[Gk*gg - nk*Sk, 2*gg*Sk - nk*Nk]
    def jac_sparsity(): pass
    r = solve_ivp(f, (t0, xs[-1]), np.r_[Gk*2*eps*t0, Gk*4*eps**2*t0**2], t_eval=xs, rtol=1e-9, atol=1e-30, method='LSODA')
    S_, N_ = r.y[:n].sum(0), r.y[n:].sum(0); D = np.hypot(N_, 2*S_)
    Dx = sum(Gk[k]*2*eps*(-np.expm1(-(nk[k] - 2*eps)*xs)/(nk[k] - 2*eps) - np.expm1(-(nk[k] + 2*eps)*xs)/(nk[k] + 2*eps)) for k in range(n))
    return D/Dx, eps*D/S_
xs = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
for a in (0.5, 0.25, 0.75):
    R, gg = numerics(a, xs); pred = 1 + float(c2.subs(al, a))*xs**2 + float(c4n[sp.nsimplify(a)])*xs**4
    print(f'alpha = {a}: x =', xs.tolist()); print('   ratio numeric:', np.round(R, 4).tolist()); print('   series       :', np.round(pred[:5], 4).tolist()); print('   gdot/eps0    :', np.round(gg, 3).tolist())
    out[f'num_{a}'] = dict(x=xs.tolist(), R=R.tolist(), gd=gg.tolist())
json.dump(out, open('oq_powerlaw.json', 'w'))
