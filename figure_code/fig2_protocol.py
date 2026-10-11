"""
Figure 2 -- the rate-controlled shear protocol (Sec. IV of the manuscript).

Protocol, Eq. (19):  the shear rate is adjusted at every instant so that the projected extension rate stays constant,
    gdot(t) s12(t)/Ds(t) = eps0,    Ds = sqrt(N1^2 + 4 s12^2),   gdot(0+) = 2 eps0.
For every single-tensor model of the class the principal stress difference Ds(t) then equals the tensile stress
difference s11 - s22 in start-up of planar extension at the constant rate eps0, at all times.

Panel (a): Ds(t)/(G tau eps0) under the protocol (symbols) and s11 - s22 in start-up of planar extension (lines),
           against Hencky strain eps0 t, for UCM (eps0 tau = 1), linear PTT, Giesekus, FENE-P (eps0 tau = 2) and the
           two-mode Maxwell fluid (eps0 taubar = 0.2), which lies outside the class.
Panel (b): shear strain gamma(t) needed, compared with exponential shear gamma = 2 sinh(eps0 t).

Run:  python3 fig2_protocol.py  (about 1 min)
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
from common import (ucm, lptt, giesekus, fenep, L_shear, L_ext, setup_matplotlib, panel_label, save_json, load_json,
                    PALETTE, INK, MUTE, FIGS, TWO_MODE, Checks)

# shear-plane components carried for each mode: A11, A12, A22, A33 (A13 = A23 = 0 in shear)
IDX = [(0, 0), (0, 1), (1, 1), (2, 2)]


def unpack(v):
    A = np.zeros((3, 3))
    A[0, 0], A[0, 1], A[1, 1], A[2, 2] = v
    A[1, 0] = v[1]
    return A


def protocol(model, eps0, T, modes):
    """Integrate the shear protocol for a fluid made of modes [(eta_k, tau_k)] of a given single-tensor model.
    Mode k obeys dA_k/dt = rhs(A_k, L tau_k)/tau_k and contributes (eta_k/tau_k) stress(A_k).
    The last state variable is the accumulated shear strain gamma."""
    n = len(modes)

    def total_stress(y):
        return sum(h / tk * model['stress'](unpack(y[4 * k:4 * k + 4])) for k, (h, tk) in enumerate(modes))

    def rate(y):
        s = total_stress(y)
        S, N1 = s[0, 1], s[0, 0] - s[1, 1]
        return 2.0 * eps0 if S < 1e-12 else eps0 * np.hypot(N1, 2.0 * S) / S

    def f(t, y):
        g = rate(y)
        out = []
        for k, (h, tk) in enumerate(modes):
            dA = model['rhs'](unpack(y[4 * k:4 * k + 4]), L_shear(g * tk)) / tk
            out += [dA[i] for i in IDX]
        return out + [g]

    y0 = np.concatenate([[model['A0'][i] for i in IDX]] * n + [[0.0]])
    sol = solve_ivp(f, [0, T], y0, method='Radau', rtol=1e-10, atol=1e-12, dense_output=True)
    return sol, total_stress, rate


def startup_planar(model, eps0, T, modes):
    f = lambda t, y: np.concatenate([np.diag(model['rhs'](np.diag(y[3 * k:3 * k + 3]), L_ext(eps0 * tk))) / tk
                                     for k, (h, tk) in enumerate(modes)])
    y0 = np.concatenate([np.diag(model['A0'])] * len(modes))
    sol = solve_ivp(f, [0, T], y0, method='Radau', rtol=1e-10, atol=1e-12, dense_output=True)

    def tensile(y):
        out = 0.0
        for k, (h, tk) in enumerate(modes):
            s = model['stress'](np.diag(y[3 * k:3 * k + 3]))
            out += h / tk * (s[0, 0] - s[1, 1])
        return out
    return sol, tensile


CASES = [  # key, model, eps0, final time, modes [(eta_k/eta0, tau_k/tau)]
    ('UCM', ucm(), 1.0, 3.5, [(1.0, 1.0)]),
    ('linear PTT', lptt(0.1), 2.0, 3.0, [(1.0, 1.0)]),
    ('Giesekus', giesekus(0.2), 2.0, 3.0, [(1.0, 1.0)]),
    ('FENE-P', fenep(50.0), 2.0, 3.0, [(1.0, 1.0)]),
    ('two-mode UCM', ucm(), 0.2, 50.0, TWO_MODE),
]


def compute():
    out = {}
    for key, m, eps0, T, modes in CASES:
        sol, stress, rate = protocol(m, eps0, T, modes)
        ext, tensile = startup_planar(m, eps0, T, modes)
        rows = []
        for t in np.linspace(T / 400, T, 400):
            y = sol.sol(t)
            s = stress(y)
            Ds = np.hypot(s[0, 0] - s[1, 1], 2 * s[0, 1])
            rows.append([t, eps0 * t, Ds / eps0, tensile(ext.sol(t)) / eps0, y[-1], rate(y)])
        rows = np.array(rows)
        out[key] = dict(eps0=eps0, rows=rows.tolist())
        print(f'{key:14s} eps0 = {eps0}: max |protocol/planar - 1| = {np.max(abs(rows[:, 2] / rows[:, 3] - 1)):.1e}; '
              f'shear strain at the end = {rows[-1, 4]:.2f} (Hencky strain {rows[-1, 1]:.2f}; 2 sinh = {2 * np.sinh(rows[-1, 1]):.1f})')
    save_json('fig2.json', out)
    return out


def plot(d):
    plt = setup_matplotlib()
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.75), constrained_layout=True)
    keys = [('UCM', r'UCM, $\dot\epsilon_0\tau=1$', PALETTE[0], 'o'), ('linear PTT', r'PTT, $\dot\epsilon_0\tau=2$', PALETTE[1], 's'),
            ('Giesekus', r'Giesekus, $\dot\epsilon_0\tau=2$', PALETTE[2], '^'), ('FENE-P', r'FENE-P, $\dot\epsilon_0\lambda_H=2$', PALETTE[3], 'D')]
    ax = axs[0]
    for k, lb, c, mk in keys:
        r = np.array(d[k]['rows'])
        ax.semilogy(r[:, 1], r[:, 3], color=c, lw=1.1)
        ax.semilogy(r[::20, 1], r[::20, 2], mk, mfc='none', mec=c, ms=4.2, mew=0.9, label=lb)
    r = np.array(d['two-mode UCM']['rows'])
    r = r[r[:, 1] <= 2.41]                             # plotted range: Hencky strain up to 2.4
    ax.semilogy(r[:, 1], r[:, 3], color=MUTE, lw=1.1, ls='--')
    ax.semilogy(r[::4, 1], r[::4, 2], 'x', color=MUTE, ms=4.5, mew=0.9, label=r'two-mode UCM, $\dot\epsilon_0\bar\tau=0.2$')
    ax.set_xlabel(r'Hencky strain $\dot\epsilon_0t$')
    ax.set_ylabel(r'$\Delta\sigma(t)/G\tau\dot\epsilon_0$')
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0.3, 3e3)
    ax.legend(loc='upper left', fontsize=6.3)
    panel_label(ax, '(a)', 'lower right')
    ax = axs[1]
    for k, lb, c, mk in keys:
        r = np.array(d[k]['rows'])
        ax.semilogy(r[:, 1], r[:, 4], color=c, label=lb.split(',')[0])
    E = np.linspace(0.05, 6, 100)
    ax.semilogy(E, 2 * np.sinh(E), color=INK, ls=':', lw=1.0, label=r'exponential shear, $2\sinh(\dot\epsilon_0t)$')
    ax.set_xlabel(r'Hencky strain $\dot\epsilon_0t$')
    ax.set_ylabel(r'shear strain $\gamma(t)$')
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0.1, 600)
    ax.legend(loc='upper left', fontsize=6.3)
    panel_label(ax, '(b)', 'lower right')
    fig.savefig(f'{FIGS}/fig2_protocol.pdf')
    plt.close(fig)


def check(d):
    ck = Checks('fig2')
    worst = max(np.max(abs(np.array(d[k]['rows'])[:, 2] / np.array(d[k]['rows'])[:, 3] - 1))
                for k in ['UCM', 'linear PTT', 'Giesekus', 'FENE-P'])
    ck.add('protocol = start-up of planar extension for the class (better than 1e-10)', '0', worst, tol=1e-10)
    for k, q in [('linear PTT', '36.4'), ('Giesekus', '42.5'), ('FENE-P', '62.1')]:
        ck.add(f'shear strain at Hencky strain 6, {k}, eps0 tau = 2', q, np.array(d[k]['rows'])[-1, 4])
    ck.add('exponential shear at Hencky strain 6', '403', 2 * np.sinh(6.0))
    r = np.array(d['two-mode UCM']['rows'])
    ratio = lambda t: np.interp(t, r[:, 0], r[:, 2] / r[:, 3])
    for t, q in [(1, '0.999'), (3, '0.979'), (10, '0.86')]:
        ck.add(f'two-mode fluid, eps0 taubar = 0.2: shear/planar at t = {t} taubar', q, ratio(t))
    # steady limit, Eq. (multi): composite/true at eps = 0.2
    e = 0.2
    steady = (4 / (1 - 4 * e * e)) / sum(4 * h / (1 - 4 * t * t * e * e) for h, t in TWO_MODE)
    ck.add('two-mode fluid: steady limit of the ratio', '0.77', steady)
    ck.add('two-mode fluid: ratio at t = 50 taubar approaches the steady limit', f'{steady:.2f}', ratio(50))
    mono = np.all(np.diff(r[:, 2] / r[:, 3]) <= 1e-12)
    ck.add('two-mode fluid: ratio decreases monotonically (1 = yes)', '1', float(mono))
    return ck.save()


if __name__ == '__main__':
    d = load_json('fig2.json') if '--plot' in sys.argv else compute()
    plot(d)
    check(d)
