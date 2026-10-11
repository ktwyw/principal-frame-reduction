"""Necessary-and-sufficient condition: tests in three-dimensional flows.  Giesekus base (alpha = 0.2); each added term has coefficient k."""
import numpy as np, json
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from lib import sym, IU, stress_model
k = 0.2; base = lambda A, D: A + 0.2*A@A
def DA(A, D):                         # part of D coaxial with A: sum_i P_i D P_i
    w, V = np.linalg.eigh(A); return V@np.diag(np.diag(V.T@D@V))@V.T
dev = lambda X: X - np.trace(X)/3*np.eye(3)
models = {
 'spin $\\bm W+\\kappa(\\bm A\\!\\cdot\\!\\bm D-\\bm D\\!\\cdot\\!\\bm A)$': (lambda A, D: base(A, D) - k*(2*A@D@A - A@A@D - D@A@A), True),
 '$[\\bm A,[\\bm A,\\bm D^2]]$': (lambda A, D: base(A, D) - k*(2*A@D@D@A - A@A@D@D - D@D@A@A), True),
 '$(\\tr\\bm D^2)\\,[\\bm A,[\\bm A,\\bm D]]$': (lambda A, D: base(A, D) - k*np.trace(D@D)*(2*A@D@A - A@A@D - D@A@A), True),
 '$\\bm D_{\\!A}^2$ (quadratic in $\\bm D$)': (lambda A, D: base(A, D) - k*DA(A, D)@DA(A, D), True),
 'relaxation rate $1+\\kappa\\tr\\bm D_{\\!A}^2$': (lambda A, D: (1 + k*np.trace(DA(A, D)@DA(A, D)))*base(A, D), True),
 '$\\bm A\\!\\cdot\\!\\bm D^2+\\bm D^2\\!\\cdot\\!\\bm A$': (lambda A, D: base(A, D) + k*(A@D@D + D@D@A), False),
 'relaxation rate $1+\\kappa\\tr\\bm D^2$': (lambda A, D: (1 + k*np.trace(D@D))*base(A, D), False)}
flows = {'I': np.array([[-.15, 2., 0], [0, -.15, 0], [0, 0, .3]]), 'II': np.array([[.2, 1.5, .7], [0, -.3, .9], [.4, 0, .1]])}
def steady(m, L, A0, T=80.0):
    f = lambda t, v: m['rhs'](sym(v), L)[IU]
    sol = solve_ivp(f, [0, T], A0[IU], method='Radau', rtol=1e-10, atol=1e-13)
    v = fsolve(lambda v: f(0, v), sol.y[:, -1], xtol=1e-14); assert np.max(abs(f(0, v))) < 1e-9, np.max(abs(f(0, v))); return sym(v)
mb = stress_model(base); out = {}
for name, (R, inside) in models.items():
    m = stress_model(R); worst = 0
    for fl, L in flows.items():
        D = (L + L.T)/2; A = steady(m, L, steady(mb, L, np.zeros((3, 3)))); a, V = np.linalg.eigh(A); d = np.diag(V.T@D@V)
        Ld = np.diag(d); X = steady(m, Ld, steady(mb, Ld, np.zeros((3, 3)))); x = np.sort(np.linalg.eigvalsh(X))
        worst = max(worst, np.max(abs(a - x))/np.max(abs(x)))
    out[name] = (worst, inside); print(f'{name:60s} inside={inside}  mismatch {worst:.1e}', flush=True)
json.dump(out, open('theory2_6.json', 'w'))
