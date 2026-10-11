"""Short-time theory of the rate-controlled shear protocol for a Maxwell spectrum, plus full-time numerics."""
import sympy as sp, numpy as np, json
from scipy.integrate import solve_ivp
t, e0 = sp.symbols('t epsilon_0', positive=True); G, nu = sp.symbols('G nu', positive=True)
NORD = 5; g = sp.symbols('g0:%d' % NORD); gd = sum(g[i]*t**i for i in range(NORD)); ms = sp.symbols('m0:8')
def lin_series(src, nmax):       # y' = src - nu y, y(0)=0, power series
    src = sp.expand(src); c = [0]*(nmax + 2)
    for i in range(nmax + 1): c[i + 1] = sp.expand((src.coeff(t, i) - nu*c[i])/(i + 1))
    return sum(c[i]*t**i for i in range(nmax + 2))
def mom(expr): expr = sp.expand(expr); return sum(expr.coeff(G, 1).coeff(nu, j)*ms[j] for j in range(8))
S = lin_series(G*gd, NORD); N1 = lin_series(sp.expand(2*gd*S), NORD)
St, Nt = mom(S), mom(N1); Sh, Nh = sp.expand(St/t), sp.expand(Nt/t**2)
con = sp.expand(gd**2*Sh**2 - e0**2*(t**2*Nh**2 + 4*Sh**2)); sol = {g[0]: 2*e0}
for i in range(1, NORD):
    eq = con.coeff(t, i).subs(sol); sol[g[i]] = sp.factor(sp.solve(eq, g[i])[0]); 
for i in range(4): print(f'  g{i} =', sol[g[i]])
# exact planar extension: per mode sigma11 - sigma22
def lin2(lam, src, nmax):          # y' = lam*y + src (constants), y(0)=0
    c = [0]*(nmax + 2)
    for i in range(nmax + 1): c[i + 1] = sp.expand(((src if i == 0 else 0) + lam*c[i])/(i + 1))
    return sum(c[i]*t**i for i in range(nmax + 2))
Dext = mom(lin2(2*e0 - nu, 2*G*e0, NORD) - lin2(-2*e0 - nu, -2*G*e0, NORD))
Dsh = sp.sqrt(Nt**2 + 4*St**2).subs(sol)
ratio = sp.series(sp.simplify(Dsh/t)/sp.expand(Dext/t), t, 0, 5).removeO(); ratio = sp.collect(sp.expand(ratio), t, sp.factor)
print('  Delta_shear/Delta_ext =', ratio)
out = dict(g=[str(sol[g[i]]) for i in range(4)], ratio=str(ratio))
# strain accumulated
print('  gamma(t) =', sp.collect(sp.expand(sp.integrate(sum(sol[g[i]]*t**i for i in range(4)), t)), t, sp.factor))
# numerics: two-mode fluid, full time course
def run(modes, eps, T):
    Gk = np.array([h/tk for h, tk in modes]); nk = np.array([1/tk for h, tk in modes]); n = len(modes); t0 = 1e-6
    def f(tt, y):
        Sk, Nk = y[:n], y[n:]; S_, N_ = Sk.sum(), Nk.sum(); gdt = eps*np.sqrt(N_**2 + 4*S_**2)/S_
        return np.r_[Gk*gdt - nk*Sk, 2*gdt*Sk - nk*Nk]
    y0 = np.r_[Gk*2*eps*t0, Gk*4*eps**2*t0**2]; tt = np.linspace(t0, T, 400)
    r = solve_ivp(f, (t0, T), y0, t_eval=tt, rtol=1e-10, atol=1e-13, method='LSODA'); S_, N_ = r.y[:n].sum(0), r.y[n:].sum(0)
    Dsh = np.sqrt(N_**2 + 4*S_**2); gdt = eps*Dsh/S_
    Dx = sum(Gk[k]*2*eps*((1 - np.exp(-(nk[k] - 2*eps)*tt))/(nk[k] - 2*eps) + (1 - np.exp(-(nk[k] + 2*eps)*tt))/(nk[k] + 2*eps)) for k in range(n))
    return tt, Dsh, Dx, gdt
modes = [(.5, .2), (.5, 1.8)]; mm = [sum(h/tk*(1/tk)**j for h, tk in modes) for j in range(8)]
for eps in (0.1, 0.2, 0.25):
    tt, Dsh, Dx, gdt = run(modes, eps, 30.0)
    pr = sp.lambdify(t, ratio.subs({ms[j]: mm[j] for j in range(8)}).subs(e0, eps))
    print(f'  eps0 = {eps}: ratio shear/ext at t = 0.1, 0.3, 1, 3, 10, 30:', [round(float(np.interp(q, tt, Dsh/Dx)), 4) for q in (0.1, 0.3, 1, 3, 10, 30)],
          '| series at 0.1, 0.3, 1:', [round(float(pr(q)), 4) for q in (0.1, 0.3, 1)], '| min ratio', round(float((Dsh/Dx).min()), 4))
    out[f'num_{eps}'] = dict(t=tt.tolist(), Dsh=Dsh.tolist(), Dx=Dx.tolist(), gd=gdt.tolist())
json.dump(out, open('theory2_3.json', 'w'))
