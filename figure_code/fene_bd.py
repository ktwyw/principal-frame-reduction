"""
fene_bd.py -- Brownian dynamics of FENE dumbbells WITHOUT closure (Sec. VI of the manuscript).

Model (units: lambda_H = 1, kT/H = 1, nkT = 1)
    dQ = [ kappa.Q - Q / (2 (1 - Q^2/b)) ] dt + dW_t ,       kappa = L (velocity gradient),
    polymer stress  tau_p = < Q Q / (1 - Q^2/b) > - I .
Integrator: semi-implicit predictor-corrector of Ottinger (1996, Sec. 4.3.2):
    predictor  Q* = Q + [kappa.Q - F(Q)/2] dt + dW,
    corrector  [1 + dt/(4 (1 - |Q'|^2/b))] Q' = Q + [kappa.(Q* + Q) - F(Q)]/2 dt + dW ,
the corrector being a scalar equation for |Q'| along the direction of the right-hand side, solved by a monotone
Newton iteration that never leaves the physical range |Q'| < sqrt(b) (validated against bisection in `test()`).

Averages: equilibrate teq, then average over tav in nblk blocks; standard errors are standard errors of block means
(successive blocks are correlated, so they somewhat underestimate the true uncertainty).

Usage
    python3 fene_bd.py test                  check the corrector root-finder against bisection
    python3 fene_bd.py run  OUT.json         run the full set of cases of Table 5 / Fig. 5 (hours on 2 cores)
"""
import json
import sys

import numpy as np


def corrector_root(Rn, b, dt, nit=12):
    """Solve x (1 + dt/(4 (1 - x^2/b))) = Rn for 0 <= x < sqrt(b) (monotone in x), vectorised Newton iteration."""
    a = dt / 4
    sb = np.sqrt(b)
    eta = np.minimum(0.5, a / (4 * np.maximum(Rn / sb, 1e-300)))
    x = np.minimum(Rn, sb * (1 - eta))           # starting point below the root, inside the domain
    for _ in range(nit):
        u = 1 - x * x / b
        f = x * (1 + a / u) - Rn
        fp = 1 + a * (1 + x * x / b) / (u * u)
        x = x - f / fp
    return x


def run(kappa, b, N=30000, dt=0.004, teq=20.0, tav=30.0, seed=0, nblk=15):
    """Simulate N independent dumbbells in the homogeneous flow kappa. Returns block averages of
    [tau_12, tau_11 + 1, tau_22 + 1, tau_33 + 1] (the identity cancels in all stress differences)."""
    rng = np.random.default_rng(seed)
    K = np.asarray(kappa, float)
    Q = rng.standard_normal((N, 3))
    r = np.linalg.norm(Q, axis=1)
    bad = r * r > 0.9 * b                         # start inside the allowed ball
    Q[bad] *= (np.sqrt(0.5 * b) / r[bad])[:, None]
    nst = int(tav / dt)
    per = nst // nblk
    acc = np.zeros((nblk, 4))
    sq = np.sqrt(dt)
    for i in range(-int(teq / dt), per * nblk):
        dW = rng.standard_normal((N, 3)) * sq
        KQ = Q @ K.T
        FQ = Q / (1 - np.einsum('ij,ij->i', Q, Q) / b)[:, None]
        Qp = Q + (KQ - 0.5 * FQ) * dt + dW                                    # predictor
        rp = np.einsum('ij,ij->i', Qp, Qp)
        bad = rp >= b * 0.999
        if bad.any():
            Qp[bad] *= (np.sqrt(0.999 * b / rp[bad]))[:, None]
        rhs = Q + 0.5 * (Qp @ K.T + KQ - 0.5 * FQ) * dt + dW                  # corrector
        Rn = np.sqrt(np.einsum('ij,ij->i', rhs, rhs))
        Q = rhs * (corrector_root(Rn, b, dt) / Rn)[:, None]
        if i >= 0:
            w = 1 / (1 - np.einsum('ij,ij->i', Q, Q) / b)
            acc[i // per] += [np.mean(w * Q[:, 0] * Q[:, 1]), np.mean(w * Q[:, 0]**2),
                              np.mean(w * Q[:, 1]**2), np.mean(w * Q[:, 2]**2)]
    return acc / per


def shear(g, b, **kw):
    """Steady shear at Wi = g: viscosity, N1, N2 and the composite (eps_eff, eta_[P1], eta_[P2]) with errors."""
    a = run([[0, g, 0], [0, 0, 0], [0, 0, 0]], b, **kw)
    S, N1, N2 = a[:, 0], a[:, 1] - a[:, 2], a[:, 2] - a[:, 3]

    def comp(S, N1, N2):
        D = np.hypot(N1, 2 * S)
        e = g * S / D
        return e, D / e, (D - N1 - 2 * N2) / (2 * e)
    m = comp(S.mean(), N1.mean(), N2.mean())
    blk = np.array([comp(*x) for x in zip(S, N1, N2)])
    n = np.sqrt(len(S))
    return dict(g=g, S=S.mean(), eta=S.mean() / g, N1=N1.mean(), N1se=N1.std(ddof=1) / n, etase=S.std(ddof=1) / n / g,
                N2=N2.mean(), N2se=N2.std(ddof=1) / n, eps=m[0], P1=m[1], P2=m[2], se=list(blk.std(0, ddof=1) / n))


def ext(e, b, **kw):
    """Steady planar extension at rate e: eta_P1, eta_P2 with errors."""
    a = run(np.diag([e, -e, 0.0]), b, **kw)
    P1 = (a[:, 1] - a[:, 2]) / e
    P2 = (a[:, 3] - a[:, 2]) / e
    n = np.sqrt(len(P1))
    return dict(e=e, P1=P1.mean(), P1se=P1.std(ddof=1) / n, P2=P2.mean(), P2se=P2.std(ddof=1) / n)


def case(args):
    """One row of Table 5: shear at Wi, then planar extension at exactly the composite rate eps_eff."""
    tag, b, g, dt_scale, seed = args
    dt = (0.004 if g <= 10 else (0.002 if g <= 30 else 0.001)) * dt_scale
    s = shear(g, b, dt=dt, seed=seed)
    x = ext(s['eps'], b, dt=(0.004 if s['eps'] < 3 else 0.002) * dt_scale, teq=40.0, seed=seed + 100)
    rec = dict(tag=tag, b=b, dts=dt_scale, seed=seed, shear=s, ext=x, ratio1=s['P1'] / x['P1'], ratio2=s['P2'] / x['P2'])
    print(f"[{tag}] b={b} Wi={g}: eta={s['eta']:.4f} N1={s['N1']:.3f} eps_eff={s['eps']:.4f} | composite {s['P1']:.3f} "
          f"true {x['P1']:.3f} ratio {rec['ratio1']:.4f}", flush=True)
    return rec


CASES = ([('curve', 50.0, g, 1.0, 1) for g in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0)] +
         [('conv-dt', 50.0, 10.0, 0.5, 11), ('conv-seed', 50.0, 10.0, 1.0, 21), ('b', 20.0, 10.0, 1.0, 1), ('b', 100.0, 10.0, 1.0, 1)])


def test():
    rng = np.random.default_rng(0)
    b, dt = 50.0, 0.004
    Rn = np.concatenate([rng.uniform(0, 12, 200000), [1e-9, np.sqrt(b), 7.0710, 40.0]])
    lo, hi = np.zeros_like(Rn), np.full_like(Rn, np.sqrt(b))
    for _ in range(60):
        x = 0.5 * (lo + hi)
        f = x * (1 + dt / (4 * (1 - x * x / b))) - Rn
        hi, lo = np.where(f > 0, x, hi), np.where(f > 0, lo, x)
    print('Newton vs bisection, max abs difference:', np.max(np.abs(corrector_root(Rn, b, dt) - 0.5 * (lo + hi))))


if __name__ == '__main__':
    if sys.argv[1:2] == ['test']:
        test()
    elif sys.argv[1:2] == ['run']:
        from multiprocessing import Pool
        with Pool() as p:
            out = p.map(case, CASES, chunksize=1)
        json.dump(out, open(sys.argv[2], 'w'), default=float)
    else:
        print(__doc__)
