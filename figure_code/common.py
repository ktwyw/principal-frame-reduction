"""
common.py -- shared code for the figures of
  Y. Wang, "Principal-frame reduction of steady viscoelastic flows: generalizing the
  equivalence of shear and planar extension" (manuscript for J. Rheol.).

Contents
  1. Plot style (vector PDF, colour + marker + line-style redundancy, serif/CM fonts).
  2. Kinematics: velocity gradients of simple shear, the planar flow family and planar extension.
  3. Single-tensor differential constitutive models written as  dA/dt = rhs(A, L),  sigma = stress(A).
  4. A robust steady-state solver for an arbitrary homogeneous flow (implicit integration from rest + Newton polish).
  5. The King-McKinley composite (Eqs. 1-3 of the manuscript) and the principal-frame projection for any planar flow.
  6. Closed-form steady material functions of single modes (linear PTT, exponential PTT, Giesekus, UCM)
     used for multimode fluids, and the K-BKZ (Wagner / PSM) integral fluid by quadrature.
  7. Bookkeeping of the numbers quoted in the manuscript (`Checks`), so every figure script can verify them.

Conventions
  L_ij = dv_i/dx_j,  D = (L + L^T)/2,  W = (L - L^T)/2.
  Simple shear: L = gdot e1 e2.  Planar extension: L = eps diag(1, -1, 0).
  Single-mode models are written in units eta0 = tau = G = 1 (so the stress tensor is sigma/G);
  a Maxwell-type mode (eta_k, tau_k) of a multimode fluid is obtained by rescaling: rates by tau_k, stresses by eta_k/tau_k.
  N1 = s11 - s22, N2 = s22 - s33,  eta_P1 = (s11 - s22)/eps,  eta_P2 = (s33 - s22)/eps.
"""
import json
import os

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import fsolve, brentq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
FIGS = os.path.join(HERE, 'figures')
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# =====================================================================================================
# 1. Plot style
# =====================================================================================================
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']   # blue, orange, green, violet (colour-blind safe set)
INK, MUTE = '#222222', '#7a7a75'


def setup_matplotlib():
    """Import matplotlib with a non-interactive backend and the journal style used for all figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 8, 'font.family': 'serif', 'mathtext.fontset': 'cm',
        'axes.linewidth': 0.6, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'legend.frameon': False, 'legend.fontsize': 7, 'lines.linewidth': 1.3,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,                    # embed TrueType fonts (journal requirement)
    })
    return plt


def panel_label(ax, text, loc='upper left'):
    """Bold panel label, e.g. '(a)', inside the axes."""
    x, ha = (0.04, 'left') if 'left' in loc else (0.96, 'right')
    y, va = (0.95, 'top') if 'upper' in loc else (0.05, 'bottom')
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9, fontweight='bold')


def save_json(name, obj):
    with open(os.path.join(DATA, name), 'w') as f:
        json.dump(obj, f, default=float)


def load_json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


# =====================================================================================================
# 2. Kinematics
# =====================================================================================================
I3 = np.eye(3)


def L_shear(gdot):
    """Simple shear, v1 = gdot x2."""
    return np.array([[0.0, gdot, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def L_planar_family(gdot, alpha):
    """Planar flow family L = gdot (e1 e2 + alpha e2 e1): alpha = -1 rotation, 0 shear, 1 planar extension."""
    return np.array([[0.0, gdot, 0.0], [alpha * gdot, 0.0, 0.0], [0.0, 0.0, 0.0]])


def L_ext(eps):
    """Planar extension at rate eps."""
    return np.diag([eps, -eps, 0.0])


# =====================================================================================================
# 3. Single-tensor constitutive models
# =====================================================================================================
# Each model is a dict with
#   rhs(A, L)  -> dA/dt for velocity gradient L,
#   stress(A)  -> polymer stress (units of G),
#   A0         -> rest state.
# The upper-convected models below are of the form of Eq. (8) of the manuscript with xi = 1.

def gordon_schowalter(R, xi=1.0):
    """Stress-form model  sigma_(GS) = 2D - R(sigma, D)  with the Gordon-Schowalter derivative
         dA/dt - W.A + A.W - xi (D.A + A.D) = 2D - R(A, D),   A = sigma/G,  eta0 = tau = 1.
    R(A, D) = A gives the Johnson-Segalman (xi < 1) / UCM (xi = 1) model."""
    def rhs(A, L):
        D = 0.5 * (L + L.T)
        W = 0.5 * (L - L.T)
        return W @ A - A @ W + xi * (D @ A + A @ D) - R(A, D) + 2.0 * D
    return dict(rhs=rhs, stress=lambda A: A, A0=np.zeros((3, 3)))


def ucm():
    return gordon_schowalter(lambda A, D: A)


def lptt(eps=0.1):
    """Linear Phan-Thien--Tanner: relaxation rate (1 + eps tr sigma)."""
    return gordon_schowalter(lambda A, D: (1.0 + eps * np.trace(A)) * A)


def eptt(eps=0.1):
    """Exponential Phan-Thien--Tanner: relaxation rate exp(eps tr sigma)."""
    return gordon_schowalter(lambda A, D: np.exp(eps * np.trace(A)) * A)


def giesekus(alpha=0.2):
    """Giesekus: sigma + alpha sigma.sigma (units G = 1)."""
    return gordon_schowalter(lambda A, D: A + alpha * A @ A)


def fenep(b=50.0):
    """FENE-P dumbbells, conformation c = <QQ> in units kT/H, time in units lambda_H:
         dc/dt = L.c + c.L^T - (f c - I),   f = 1/(1 - tr c / b),   sigma/nkT = f c - I."""
    f = lambda c: 1.0 / (1.0 - np.trace(c) / b)
    return dict(rhs=lambda c, L: L @ c + c @ L.T - (f(c) * c - I3),
                stress=lambda c: f(c) * c - I3,
                A0=I3 * b / (b + 3.0))


def rolie_poly(td=1.0, tR=0.2, beta=0.1, delta=-0.5, Lmax=10.0):
    """Rolie-Poly with finite extensibility (Likhtman & Graham 2003; Kabanemi & Hetu form of k_s),
    the parameter set used by King & McKinley (tau_d/tau_R = 5, beta = 0.1, delta = -0.5, Lmax = 10)."""
    def ks(l2):
        return (3 - l2 / Lmax**2) * (1 - 1 / Lmax**2) / ((1 - l2 / Lmax**2) * (3 - 1 / Lmax**2))

    def rhs(c, L):
        l2 = np.trace(c) / 3.0
        lam = np.sqrt(l2)
        h = 2.0 * ks(l2) / tR * (1.0 - 1.0 / lam)
        return L @ c + c @ L.T - (c - I3) / td - h * (c + beta * lam**(2 * delta) * (c - I3))
    return dict(rhs=rhs, stress=lambda c: ks(np.trace(c) / 3.0) * (c - I3), A0=I3.copy())


# =====================================================================================================
# 4. Steady state in an arbitrary homogeneous flow
# =====================================================================================================
_IU = np.triu_indices(3)


def _sym(v):
    A = np.zeros((3, 3))
    A[_IU] = v
    return A + A.T - np.diag(np.diag(A))


def steady_state(model, L, T=400.0, guess=None):
    """Steady solution of dA/dt = rhs(A, L).
    The full six-component tensor equation is integrated from rest with an implicit Runge-Kutta method
    (Radau IIA, rtol 1e-10) and the end point is polished with a Newton-type solve (residual < 1e-9).
    Starting from rest selects the branch that is reached physically; shear-free reference solutions are
    always computed independently in the same way (never from a mapped state)."""
    f = lambda t, v: model['rhs'](_sym(v), L)[_IU]
    if guess is None:
        sol = solve_ivp(f, [0, T], model['A0'][_IU], method='Radau', rtol=1e-10, atol=1e-13)
        guess = sol.y[:, -1]
    v, info, ok, _ = fsolve(lambda v: f(0, v), guess, xtol=1e-14, full_output=True)
    if ok != 1 or np.max(np.abs(f(0, v))) > 1e-9:          # not yet converged: integrate much longer
        sol = solve_ivp(f, [0, 50 * T], model['A0'][_IU], method='Radau', rtol=1e-11, atol=1e-13)
        v = fsolve(lambda v: f(0, v), sol.y[:, -1], xtol=1e-14)
    res = np.max(np.abs(f(0, v)))
    if res > 1e-8:
        raise RuntimeError(f'steady state not converged (residual {res:.1e})')
    return _sym(v)


# =====================================================================================================
# 5. Composite and projections
# =====================================================================================================
def composite_from_shear(sigma, gdot):
    """King-McKinley composite from a steady simple-shear stress tensor (Eqs. 1-3):
        eps_eff = gdot s12/Ds,  Ds = sqrt(N1^2 + 4 s12^2),
        eta_[P1] = Ds/eps_eff,  eta_[P2] = Ds (Ds - N1 - 2 N2)/(2 gdot s12)."""
    S, N1, N2 = sigma[0, 1], sigma[0, 0] - sigma[1, 1], sigma[1, 1] - sigma[2, 2]
    Ds = np.hypot(N1, 2.0 * S)
    e = gdot * S / Ds
    return e, Ds / e, (Ds - N1 - 2.0 * N2) / (2.0 * e)


def composite_from_functions(S, N1, gdot, N2=0.0):
    """Same as composite_from_shear, from shear stress S, N1 and N2 (arrays allowed)."""
    Ds = np.hypot(N1, 2.0 * S)
    e = gdot * S / Ds
    return e, Ds / e, (Ds - N1 - 2.0 * N2) / (2.0 * e)


def planar_projection(sigma, L):
    """Principal-frame projection for any planar flow: returns d1 = n+.D.n+ (n+ = major in-plane principal axis
    of the stress) and the projected viscosities (s+ - s-)/d1 and (s3 - s-)/d1."""
    D = 0.5 * (L + L.T)
    w, V = np.linalg.eigh(sigma[:2, :2])
    n = np.array([V[0, 1], V[1, 1], 0.0])
    d1 = n @ D @ n
    return d1, (w[1] - w[0]) / d1, (sigma[2, 2] - w[0]) / d1


def planar_viscosities(sigma, eps):
    return (sigma[0, 0] - sigma[1, 1]) / eps, (sigma[2, 2] - sigma[1, 1]) / eps


# =====================================================================================================
# 6. Closed-form single-mode material functions (mode viscosity eta_k, time tau_k)
# =====================================================================================================
# Each *_shear returns (s12, N1) at shear rate gdot; each *_ext returns eta_P1 at extension rate eps.
# They are checked against the full-tensor solver in selftest().

def lptt_shear(eta_k, tau_k, ep, gdot):
    """Linear PTT mode in steady shear: f = 1 + ep tr(sigma)/G solves f^3 - f^2 - 2 ep Wi^2 = 0."""
    Wi = tau_k * gdot
    if ep == 0.0:
        return eta_k * gdot, 2.0 * eta_k * tau_k * gdot**2
    f = brentq(lambda f: f**3 - f**2 - 2.0 * ep * Wi * Wi, 1.0, 1e14, xtol=1e-15, rtol=1e-15)
    return eta_k * gdot / f, 2.0 * eta_k * tau_k * gdot**2 / f**2


def lptt_ext(eta_k, tau_k, ep, eps):
    """Linear PTT mode in steady planar extension: f solves (f - 1)(f^2 - 4 w^2) = 8 ep w^2, w = tau eps;
    eta_P1 = 4 eta_k f/(f^2 - 4 w^2)."""
    w = tau_k * eps
    if ep == 0.0:
        return 4.0 * eta_k / (1.0 - 4.0 * w * w) if 2 * w < 1 else np.inf
    lo = max(1.0, 2.0 * w) * (1.0 + 1e-14)
    f = brentq(lambda f: (f - 1.0) * (f * f - 4.0 * w * w) - 8.0 * ep * w * w, lo, 2.0 * w + 11.0 + 10.0 * ep,
               xtol=1e-15, rtol=1e-15)
    return 4.0 * eta_k * f / (f * f - 4.0 * w * w)


def eptt_shear(eta_k, tau_k, ep, gdot):
    """Exponential PTT mode in steady shear: f = exp(ep tr sigma/G) solves f^2 ln f = 2 ep Wi^2."""
    Wi = tau_k * gdot
    f = brentq(lambda f: f * f * np.log(f) - 2.0 * ep * Wi * Wi, 1.0, 1e9, xtol=1e-15, rtol=1e-15) if Wi > 0 else 1.0
    return eta_k * gdot / f, 2.0 * eta_k * tau_k * gdot**2 / f**2


def eptt_ext(eta_k, tau_k, ep, eps):
    """Exponential PTT mode in planar extension: ln f (f^2 - 4 w^2) = 8 ep w^2; eta_P1 = 4 eta_k f/(f^2 - 4w^2)."""
    w = tau_k * eps
    lo = max(1.0, 2.0 * w) * (1.0 + 1e-13) + 1e-300
    f = brentq(lambda f: np.log(f) * (f * f - 4.0 * w * w) - 8.0 * ep * w * w, lo, 2.0 * w + 50.0, xtol=1e-15, rtol=1e-15)
    return 4.0 * eta_k * f / (f * f - 4.0 * w * w)


def giesekus_shear(eta_k, tau_k, al, gdot):
    """Giesekus mode in steady shear (Bird, Armstrong & Hassager 1987, Table 4.3-3)."""
    Wi = tau_k * gdot
    if Wi == 0:
        return 0.0, 0.0
    chi = np.sqrt((np.sqrt(1 + 16 * al * (1 - al) * Wi**2) - 1) / (8 * al * (1 - al) * Wi**2))
    f = (1 - chi) / (1 + (1 - 2 * al) * chi)
    eta = eta_k * (1 - f)**2 / (1 + (1 - 2 * al) * f)
    psi1 = 2 * eta_k * tau_k * f * (1 - al * f) / (Wi**2 * al * (1 - f))
    return eta * gdot, psi1 * gdot**2


def giesekus_ext(eta_k, tau_k, al, eps):
    """Giesekus mode in steady planar extension, closed form."""
    w = tau_k * eps
    if w == 0:
        return 4.0 * eta_k
    return eta_k * (np.sqrt((1 - 2 * w)**2 + 8 * al * w) - np.sqrt((1 + 2 * w)**2 - 8 * al * w) + 4 * w) / (2 * al * w)


def multimode(modes, gdot, shear_fn, ext_fn):
    """Composite of a fluid made of independent modes [(eta_k, tau_k, p_k), ...] and its true eta_P1
    at the composite rate. Returns eps_eff, eta_[P1], eta_P1(eps_eff)."""
    S = N1 = 0.0
    for (h, t, p) in modes:
        s, n = shear_fn(h, t, p, gdot)
        S += s
        N1 += n
    e, c1, _ = composite_from_functions(S, N1, gdot)
    return e, c1, sum(ext_fn(h, t, p, e) for (h, t, p) in modes)


def kbkz_psm(modes, a, gdot):
    """Separable K-BKZ (Wagner) fluid with memory sum_k (eta_k/tau_k^2) exp(-s/tau_k) and
    Papanastasiou-Scriven-Macosko damping h = 1/(1 + a(I1 - 3)).  Steady shear: I1 - 3 = (gdot s)^2;
    planar extension: I1 - 3 = 4 sinh^2(eps s), s11 - s22 = int m h 2 sinh(2 eps s) ds.
    Returns eps_eff, eta_[P1](eps_eff), eta_P1(eps_eff)."""
    h = lambda I: 1.0 / (1.0 + a * I)
    kw = dict(epsabs=1e-13, epsrel=1e-12, limit=600)
    S = sum(quad(lambda s: hk / tk**2 * np.exp(-s / tk) * h((gdot * s)**2) * gdot * s, 0, np.inf, **kw)[0] for hk, tk in modes)
    N1 = sum(quad(lambda s: hk / tk**2 * np.exp(-s / tk) * h((gdot * s)**2) * (gdot * s)**2, 0, np.inf, **kw)[0] for hk, tk in modes)
    e, c1, _ = composite_from_functions(S, N1, gdot)

    def kern(u):                         # 2 sinh(2u)/(1 + 4 a sinh^2 u), written without overflow
        e2 = np.exp(-2.0 * u)
        return (1.0 - e2 * e2) / (e2 + a * (1.0 - e2)**2)
    tr = sum(quad(lambda s: hk / tk**2 * np.exp(-s / tk) * kern(e * s), 0, np.inf, **kw)[0] for hk, tk in modes) / e
    return e, c1, tr


# the two-mode spectrum used throughout Secs. IV-V:  (eta_k/eta0, tau_k/taubar)
TWO_MODE = [(0.5, 0.2), (0.5, 1.8)]
TWO_MODE_VAR = sum(h * t * t for h, t in TWO_MODE) - sum(h * t for h, t in TWO_MODE)**2   # Var(tau) = 0.64 taubar^2


# =====================================================================================================
# 7. Checks of numbers quoted in the manuscript
# =====================================================================================================
class Checks:
    """Collects (description, value quoted in the manuscript, value computed here, tolerance)."""

    def __init__(self, figure):
        self.figure = figure
        self.rows = []

    def add(self, what, quoted, computed, tol=None, note=''):
        """`quoted` is the number as printed in the manuscript (a string, e.g. '0.76', '4.9e-4', '403').
        Default tolerance: half a unit in its last printed digit (i.e. the computed value must round to it)."""
        q, c = float(quoted), float(computed)
        if tol is None:
            mant = quoted.lower().split('e')
            dp = len(mant[0].split('.')[1]) if '.' in mant[0] else 0
            tol = 0.5 * 10**(-dp) * (10**int(mant[1]) if len(mant) > 1 else 1) * 1.0001
        ok = abs(c - q) <= tol
        self.rows.append(dict(figure=self.figure, what=what, quoted=quoted, computed=c, tol=tol, ok=bool(ok), note=note))
        print(f"  [{'OK  ' if ok else 'FAIL'}] {what}: manuscript {quoted}, computed {c:.6g} {note}")
        return ok

    def at_most(self, what, bound, computed, note=''):
        """Checks a claim of the form 'at most / within / below <bound>' (bound as printed, e.g. '0.08')."""
        c = float(computed)
        ok = c <= float(bound)
        self.rows.append(dict(figure=self.figure, what=what, quoted='<= ' + bound, computed=c, tol=0.0, ok=bool(ok), note=note))
        print(f"  [{'OK  ' if ok else 'FAIL'}] {what}: manuscript <= {bound}, computed {c:.6g} {note}")
        return ok

    def save(self):
        save_json(f'checks_{self.figure}.json', self.rows)
        return all(r['ok'] for r in self.rows)


# =====================================================================================================
# Self-test: closed-form mode functions against the full-tensor solver
# =====================================================================================================
def selftest():
    print('self-test of closed-form single-mode material functions against the full-tensor solver')
    worst = 0.0
    for name, m, sh, ex, p in [('linear PTT', lptt(0.1), lptt_shear, lptt_ext, 0.1),
                               ('exponential PTT', eptt(0.1), eptt_shear, eptt_ext, 0.1),
                               ('Giesekus', giesekus(0.2), giesekus_shear, giesekus_ext, 0.2)]:
        for g in [0.05, 0.5, 3.0]:
            s = steady_state(m, L_shear(g))
            S, N1 = sh(1.0, 1.0, p, g)
            e = 0.3 * g
            x = steady_state(m, L_ext(e), T=2000)
            r = max(abs(S / s[0, 1] - 1), abs(N1 / (s[0, 0] - s[1, 1]) - 1), abs(ex(1.0, 1.0, p, e) / ((x[0, 0] - x[1, 1]) / e) - 1))
            worst = max(worst, r)
        print(f'  {name:16s} max relative difference {worst:.1e}')
    assert worst < 1e-8, worst
    return worst


if __name__ == '__main__':
    selftest()
