"""Shared numerics: single-tensor constitutive models, steady states in arbitrary homogeneous flows,
the King-McKinley composite, and analytic multimode linear-PTT branches.  Units eta0 = tau = G = 1."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq
I3 = np.eye(3); IU = np.triu_indices(3)
def sym(v):
    A = np.zeros((3, 3)); A[IU] = v; return A + A.T - np.diag(np.diag(A))
def stress_model(R, a=1.0):
    def rhs(A, L):
        D = (L + L.T)/2; W = (L - L.T)/2
        return W@A - A@W + a*(D@A + A@D) - R(A, D) + 2*D
    return dict(rhs=rhs, stress=lambda A: A, A0=np.zeros((3, 3)))
def rolie_poly(td=1.0, tR=0.2, beta=0.1, delta=-0.5, Lmax=10.0):
    ks = lambda l2: (3 - l2/Lmax**2)*(1 - 1/Lmax**2)/((1 - l2/Lmax**2)*(3 - 1/Lmax**2))
    def rhs(c, L):
        l2 = np.trace(c)/3; lam = np.sqrt(l2); h = 2*ks(l2)/tR*(1 - 1/lam)
        return L@c + c@L.T - (c - I3)/td - h*(c + beta*lam**(2*delta)*(c - I3))
    return dict(rhs=rhs, stress=lambda c: ks(np.trace(c)/3)*(c - I3), A0=I3.copy())
def fenep(b=50.0):
    f = lambda c: 1/(1 - np.trace(c)/b)
    return dict(rhs=lambda c, L: L@c + c@L.T - (f(c)*c - I3), stress=lambda c: f(c)*c - I3, A0=I3*b/(b+3))
ucm = lambda: stress_model(lambda A, D: A)
lptt = lambda eps=0.1: stress_model(lambda A, D: (1 + eps*np.trace(A))*A)
eptt = lambda eps=0.1: stress_model(lambda A, D: np.exp(eps*np.trace(A))*A)
giesekus = lambda al=0.2: stress_model(lambda A, D: A + al*A@A)
def steady(model, L, T=400.0, guess=None):
    f = lambda t, v: model['rhs'](sym(v), L)[IU]
    if guess is None:
        sol = solve_ivp(f, [0, T], model['A0'][IU], method='Radau', rtol=1e-10, atol=1e-13); guess = sol.y[:, -1]
    v, info, ok, _ = fsolve(lambda v: f(0, v), guess, xtol=1e-14, full_output=True)
    if ok != 1 or np.max(np.abs(f(0, v))) > 1e-9:
        sol = solve_ivp(f, [0, 50*T], model['A0'][IU], method='Radau', rtol=1e-11, atol=1e-13); v = fsolve(lambda v: f(0, v), sol.y[:, -1], xtol=1e-14)
    return sym(v)
Lshear = lambda g: np.array([[0, g, 0], [0, 0, 0], [0, 0, 0.]])
Lmixed = lambda g, al: np.array([[0, g, 0], [al*g, 0, 0], [0, 0, 0.]])
Lext = lambda e: np.diag([e, -e, 0.])
def composite(s, g):
    """King-McKinley composite from a steady simple-shear stress tensor."""
    S, N1, N2 = s[0, 1], s[0, 0] - s[1, 1], s[1, 1] - s[2, 2]; Dl = np.hypot(N1, 2*S); e = g*S/Dl
    return e, Dl/e, (Dl - N1 - 2*N2)/2/e
def planar_projection(s, L):
    """Principal-frame projection for any planar flow: returns d1 = n+.D.n+, (s+ - s-)/d1, (s3 - s-)/d1."""
    D = (L + L.T)/2; w, V = np.linalg.eigh(s[:2, :2]); n = np.array([V[0, 1], V[1, 1], 0.0]); d1 = n@D@n
    return d1, (w[1] - w[0])/d1, (s[2, 2] - w[0])/d1
# ---- analytic multimode linear PTT (each mode: viscosity h, time t, parameter ep)
def lptt_shear(h, t, ep, g):
    u = t*g
    if ep == 0: return h*g, 2*h*t*g*g
    f = brentq(lambda f: f**3 - f**2 - 2*ep*u*u, 1, 1e14); return h*g/f, 2*h*t*g*g/f**2
def lptt_ext(h, t, ep, x):
    w = t*x
    if ep == 0: return 4*h/(1 - 4*w*w) if 2*w < 1 else np.inf
    f = brentq(lambda f: (f - 1)*(f*f - 4*w*w) - 8*ep*w*w, max(1, 2*w)*(1 + 1e-14), 2*w + 11 + 10*ep); return 4*h*f/(f*f - 4*w*w)
def lptt_multi(modes, g):
    S = sum(lptt_shear(*m, g)[0] for m in modes); N1 = sum(lptt_shear(*m, g)[1] for m in modes); D = np.hypot(N1, 2*S); x = g*S/D
    return x, D/x, sum(lptt_ext(*m, x) for m in modes)
