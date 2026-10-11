"""Extensions for the full paper: (1) class-membership tests (pom-pom with a scalar stretch, Oldroyd-type terms,
excluded generators); (2) O(eps^4) defect of a single-mode K-BKZ fluid; (3) rate-controlled (constant eps_eff) shear protocol."""
import numpy as np, json, sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from lib import *
out = {}
# ---------------------------------------------------------------- 1a. differential pom-pom: tensor A + scalar stretch lam
tb, ts, q = 1.0, 1/3.0, 8.0; nu = 2/q
def pp_rhs(y, L):
    A = sym(y[:6]); lam = y[6]; D = (L + L.T)/2; S = A/np.trace(A)
    dA = L@A + A@L.T - (A - I3)/tb
    dl = lam*np.sum(D*S) - (lam - 1)/ts*np.exp(nu*(lam - 1))
    return np.concatenate([dA[IU], [dl]])
def pp_steady(L, T=200):
    f = lambda t, y: pp_rhs(y, L); sol = solve_ivp(f, [0, T], np.concatenate([I3[IU], [1.0]]), method='Radau', rtol=1e-10, atol=1e-13)
    y = fsolve(lambda y: f(0, y), sol.y[:, -1], xtol=1e-14); A = sym(y[:6]); return 3*y[6]**2*A/np.trace(A), y[6]
rows = []
for g in [0.1, 0.3, 1, 3, 10, 30]:
    s, lam = pp_steady(Lshear(g)); e, P1, P2 = composite(s, g); se, lame = pp_steady(Lext(e), T=2000)
    rows.append([g, e, lam, lame, P1, (se[0, 0]-se[1, 1])/e, P2, (se[2, 2]-se[1, 1])/e])
out['pompom'] = rows
print('pom-pom (tau_b/tau_s=3, q=8): max rel diff P1, P2, stretch:', max(abs(r[4]/r[5]-1) for r in rows), max(abs(r[6]/r[7]-1) for r in rows), max(abs(r[2]/r[3]-1) for r in rows), ' max stretch', max(r[2] for r in rows))
# ---------------------------------------------------------------- 1b. other members and non-members
def test(name, m, rates, T=400):
    worst = 0
    for g in rates:
        s = m['stress'](steady(m, Lshear(g), T=T)); e, P1, P2 = composite(s, g); x = m['stress'](steady(m, Lext(e), T=20*T))
        worst = max(worst, abs(P1/((x[0, 0]-x[1, 1])/e) - 1), abs(P2/((x[2, 2]-x[1, 1])/e) - 1))
    print(f'{name:58s} max |composite/true - 1| = {worst:.2e}'); out.setdefault('members', {})[name] = worst
r1 = [0.1, 0.3, 1.0, 3.0]
test('Oldroyd-type: xi=0.8, +0.1 tr(s) D + 0.1 (s:D) I   [in]', stress_model(lambda A, D: A + 0.1*np.trace(A)*D + 0.1*np.sum(A*D)*I3, a=0.8), [0.1, 0.3, 1.0])
test('s.D + D.s and s^2.D + D.s^2 terms (psi_1, psi_2)     [in]', stress_model(lambda A, D: A + 0.05*(A@D + D@A) + 0.02*(A@A@D + D@A@A)), r1)
test('relaxation rate depending on tr(s^2 D)               [in]', stress_model(lambda A, D: (1 + 0.1*np.tanh(np.sum((A@A)*D)))*A), r1)
test('Larson partially extending: + (2 xi/3)(D:s) s        [in]', stress_model(lambda A, D: A + 0.2*np.sum(A*D)*(A + I3)), r1)
test('D.D term (Reiner-Rivlin-like), kappa = 0.3          [out]', stress_model(lambda A, D: A + 0.3*D@D), r1)
test('s.D^2 + D^2.s term, kappa = 0.3                     [out]', stress_model(lambda A, D: A + 0.3*(A@D@D + D@D@A)), [0.1, 0.3, 1.0])
test('coefficient depending on tr(D^2): White-Metzner     [out]', stress_model(lambda A, D: A*(1 + np.sqrt(2*np.sum(D*D)))), r1)   # tau = 1/(1+gdot), modulus not constant here
test('coefficient depending on tr(s D^2)                  [out]', stress_model(lambda A, D: (1 + 0.3*np.tanh(np.sum(A*(D@D))))*A), r1)
# ---------------------------------------------------------------- 2. K-BKZ single mode: O(eps^4) defect, analytic
a, g, e, s = sp.symbols('a gamma epsilon s', positive=True)
hser = lambda I: 1 - a*I + a**2*I**2            # h = 1/(1 + a I) to second order in I
S = sp.integrate(sp.exp(-s)*sp.expand(hser(g**2*s**2)*g*s), (s, 0, sp.oo)); N1 = sp.integrate(sp.exp(-s)*sp.expand(hser(g**2*s**2)*g**2*s**2), (s, 0, sp.oo))
I1 = sp.series(4*sp.sinh(e*s)**2, e, 0, 7).removeO()
integrand = sp.series(sp.expand(hser(I1)*sp.series(2*sp.sinh(2*e*s), e, 0, 8).removeO()), e, 0, 6).removeO()
true = sp.expand(sp.integrate(sp.exp(-s)*sp.expand(integrand), (s, 0, sp.oo))/e)
Dl = sp.sqrt(N1**2 + 4*S**2); eeff = sp.series(g*S/Dl, g, 0, 6).removeO(); c3, c5 = sp.symbols('c3 c5')
gs = 2*e + c3*e**3 + c5*e**5; eq = sp.expand(sp.series(eeff.subs(g, gs), e, 0, 6).removeO() - e)
sol = sp.solve([eq.coeff(e, 3), eq.coeff(e, 5)], [c3, c5], dict=True)[0]
comp = sp.series((Dl**2/(g*S)).subs(g, gs.subs(sol)), e, 0, 6).removeO()
d4 = sp.factor(sp.expand(comp - true).coeff(e, 4)); print('K-BKZ single mode: eta_[P1] - eta_P1 =', sp.factor(sp.expand(comp - true).coeff(e, 2)), '* eps^2 +', d4, '* eps^4;  value at a=0.05:', float(d4.subs(a, 0.05)))
out['kbkz_d4'] = str(d4)
# ---------------------------------------------------------------- 3. rate-controlled shear protocol
IDX = [(0, 0), (0, 1), (1, 1), (2, 2)]
def unp(v):
    A = np.zeros((3, 3)); A[0, 0], A[0, 1], A[1, 1], A[2, 2] = v; A[1, 0] = v[1]; return A
def feedback(model, eps0, T, modes):
    n = len(modes)
    def f(t, y):
        As = [unp(y[4*k:4*k+4]) for k in range(n)]; sig = sum(h/tk*model['stress'](A) for (h, tk), A in zip(modes, As))
        S, N1 = sig[0, 1], sig[0, 0] - sig[1, 1]; Dl = np.hypot(N1, 2*S); g = 2*eps0 if S < 1e-12 else eps0*Dl/S
        o = []
        for (h, tk), A in zip(modes, As): o += [model['rhs'](A, Lshear(g*tk))[i]/tk for i in IDX]
        return o + [g]
    y0 = np.concatenate([[model['A0'][i] for i in IDX]]*n + [[0.0]])
    return solve_ivp(f, [0, T], y0, method='Radau', rtol=1e-10, atol=1e-12, dense_output=True)
def ext_start(model, eps0, T, modes):
    f = lambda t, y: np.concatenate([np.diag(model['rhs'](np.diag(y[3*k:3*k+3]), Lext(eps0*tk)))/tk for k, (h, tk) in enumerate(modes)])
    return solve_ivp(f, [0, T], np.concatenate([np.diag(model['A0'])]*len(modes)), method='Radau', rtol=1e-10, atol=1e-12, dense_output=True)
prot = {}
for key, m, eps0, T, modes in [('UCM, Wi=1', ucm(), 1.0, 3.5, [(1., 1.)]), ('linear PTT, Wi=2', lptt(), 2.0, 3.0, [(1., 1.)]), ('Giesekus, Wi=2', giesekus(0.2), 2.0, 3.0, [(1., 1.)]),
                               ('FENE-P (b=50), Wi=2', fenep(50.0), 2.0, 3.0, [(1., 1.)]), ('two-mode UCM, Wi=0.2', ucm(), 0.2, 12.0, [(.5, .2), (.5, 1.8)]), ('two-mode UCM, Wi=0.4', ucm(), 0.4, 8.0, [(.5, .2), (.5, 1.8)])]:
    sol = feedback(m, eps0, T, modes); ex = ext_start(m, eps0, T, modes); ts_ = np.linspace(T/60, T, 60); rec = []
    for t in ts_:
        y = sol.sol(t); sig = sum(h/tk*m['stress'](unp(y[4*k:4*k+4])) for k, (h, tk) in enumerate(modes)); Dl = np.hypot(sig[0, 0]-sig[1, 1], 2*sig[0, 1]); z = ex.sol(t)
        tr = sum(h/tk*(m['stress'](np.diag(z[3*k:3*k+3]))[0, 0] - m['stress'](np.diag(z[3*k:3*k+3]))[1, 1]) for k, (h, tk) in enumerate(modes))
        rec.append([t, eps0*t, Dl/eps0, tr/eps0, y[-1], eps0*Dl/sig[0, 1]])
    rec = np.array(rec); prot[key] = rec.tolist()
    print(f'{key:24s} max |ratio-1| = {np.max(abs(rec[:,2]/rec[:,3]-1)):.2e}; final Hencky strain {rec[-1,1]:.1f} needs shear strain {rec[-1,4]:.1f} (2 sinh: {2*np.sinh(rec[-1,1]):.1f}); final shear rate/eps0 = {rec[-1,5]/eps0:.1f}')
out['protocol'] = prot
json.dump(out, open('extend.json', 'w'), default=float)
