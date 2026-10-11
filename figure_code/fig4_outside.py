"""
Figure 4 -- ratio of the composite to the true planar viscosity for fluids outside the class (Sec. V C-D).

Curves (x axis: eps_eff taubar; taubar = viscosity-weighted mean relaxation time, the zero-rate time for White-Metzner):
  * White-Metzner, constant modulus, tau = tau0/(1 + tau0 sqrt(2 tr D^2))  [closed form; tau depends on tr D^2]
  * single-mode K-BKZ with PSM damping, a = 0.05 and 0.2               [quadrature]
  * two-mode linear PTT, common eps = 0.1, spectrum (0.5, 0.2), (0.5, 1.8)
  * two-mode linear PTT with unequal eps: (p_k, tau_k/taubar, eps_k) = (0.47, 2.03, 0.080), (0.53, 0.094, 0.0015)
  * Oldroyd-B, eta_s/eta0 = 0.1 (composite applied to the total stress)
  * Rouse chain (Maxwell modes)
Also computed: the multimode linear-PTT scans quoted in Sec. V D and the high-rate plateau factor 0.890.

Run:  python3 fig4_outside.py             (about 5 min; the PTT scans take most of it)
      python3 fig4_outside.py --no-scan   (skip the scans)
"""
import sys
import numpy as np
from scipy.special import zeta
from common import (lptt_shear, lptt_ext, multimode, kbkz_psm, setup_matplotlib, save_json, load_json,
                    PALETTE, INK, MUTE, FIGS, TWO_MODE, Checks)

# two linear-PTT modes with unequal eps (found by a random search for a large overshoot of the composite)
UNEQUAL = [(0.4692560315550124, 2.422716318685524, 0.08047825511831319),
           (0.5307439684449876, 0.11263365816017955, 0.001533190976120176)]


def white_metzner(gdot_tau0):
    """White-Metzner with constant modulus G = 1, tau = 1/(1 + sqrt(2 tr D^2)) (tau0 = 1).
    Shear: sqrt(2 tr D^2) = gdot, tau_s = 1/(1 + gdot); eta = tau_s, N1/2s12 = tau_s gdot.
    Planar extension at eps: sqrt(2 tr D^2) = 2 eps, tau_e = 1/(1 + 2 eps), eta_P1 = 4 tau_e/(1 - 4 tau_e^2 eps^2)."""
    g = gdot_tau0
    ts = 1 / (1 + g)
    r = ts * g
    e = g / (2 * np.sqrt(1 + r * r))
    comp = 4 * ts * (1 + r * r)
    te = 1 / (1 + 2 * e)
    true = 4 * te / (1 - 4 * te * te * e * e)
    return e, comp, true


def compute(scan=True):
    out = {'curves': {}}
    g = np.logspace(-2, 2.5, 120)
    e, c, t = white_metzner(g)
    out['curves']['White-Metzner (one tensor)'] = np.c_[e, c / t].tolist()
    for a in [0.05, 0.2]:
        out['curves'][f'K-BKZ, one mode, a = {a}'] = [[x, cc / tr] for x, cc, tr in
                                                      [kbkz_psm([(1.0, 1.0)], a, gd) for gd in np.logspace(-1.5, 2.0, 60)]]
    out['curves']['two-mode PTT, common eps'] = [[x, cc / tr] for x, cc, tr in
                                                 [multimode([(h, t_, 0.1) for h, t_ in TWO_MODE], gd, lptt_shear, lptt_ext)
                                                  for gd in np.logspace(-2, 3, 100)]]
    tb = sum(h * t_ for h, t_, _ in UNEQUAL)
    out['curves']['two-mode PTT, unequal eps'] = [[x * tb, cc / tr] for x, cc, tr in
                                                  [multimode(UNEQUAL, gd, lptt_shear, lptt_ext) for gd in np.logspace(-2, 2.6, 160)]]
    w = np.linspace(0.005, 0.4995, 200)
    bs = 0.1                                                    # Oldroyd-B, eta_s/eta0
    comp = 4 / (1 - 4 * (1 - bs)**2 * w * w)                    # composite of the total stress at eps tau = w
    true = 4 * bs + 4 * (1 - bs) / (1 - 4 * w * w)
    out['curves']['Oldroyd-B, eta_s/eta0 = 0.1'] = np.c_[w * (1 - bs), comp / true].tolist()   # taubar = (1 - bs) tau
    k = 2.0
    tbR = zeta(2 * k) / zeta(k)
    tt = np.arange(1, 200001.0)**-k
    rouse = []
    for x in np.linspace(0.005, 0.498, 120):                    # x = eps tau_1
        if 2 * tbR * x < 1:
            rouse.append([x * tbR, (4 / (1 - 4 * tbR**2 * x * x)) / (4 * (tt / (1 - 4 * tt * tt * x * x)).sum() / tt.sum())])
    out['curves']['Rouse chain'] = rouse
    # numbers quoted in the text
    out['WM_at_5'] = white_metzner(5.0)
    # high-rate plateau of the common-eps PTT pair: Cauchy-Schwarz factor
    p = np.array([h for h, t_ in TWO_MODE])
    tk = np.array([t_ for h, t_ in TWO_MODE])
    out['PTT_plateau_factor'] = (p * tk**(-1 / 3)).sum()**2 / (p * tk**(-2 / 3)).sum()
    x, cc, tr = multimode([(h, t_, 0.1) for h, t_ in TWO_MODE], 1e6, lptt_shear, lptt_ext)
    out['PTT_plateau_numeric'] = cc / tr
    if scan:
        out['scan'] = ptt_scan()
    save_json('fig4.json', out)
    return out


def ptt_scan(n_spectra=600, seed=11):
    """Random two- to four-mode linear-PTT spectra with a common eps: relaxation times log-uniform over two decades,
    viscosity fractions Dirichlet(1,...,1); shear rates 0.03 <= gdot taubar <= 1e4. Returns min/max of eta_[P1]/eta_P1."""
    res = {}
    for ep in (0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
        rng = np.random.default_rng(seed)
        lo, hi = 9.0, 0.0
        for _ in range(n_spectra):
            n = rng.integers(2, 5)
            tk = 10**rng.uniform(-2, 0, n)
            hk = rng.dirichlet(np.ones(n))
            tb = (hk * tk).sum()
            modes = [(h, t_ / tb, ep) for h, t_ in zip(hk, tk)]
            for g in np.logspace(-1.5, 4, 45):
                x, c, tr = multimode(modes, g, lptt_shear, lptt_ext)
                lo, hi = min(lo, c / tr), max(hi, c / tr)
        res[str(ep)] = [lo, hi]
        print(f'  PTT scan, eps = {ep}: ratio in [{lo:.3f}, {hi:.3f}]', flush=True)
    return res


def plot(d):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.5, 2.9), constrained_layout=True)
    style = {'White-Metzner (one tensor)': (PALETTE[0], '-'), 'K-BKZ, one mode, a = 0.05': (PALETTE[1], '-'),
             'K-BKZ, one mode, a = 0.2': (PALETTE[1], '--'), 'two-mode PTT, common eps': (PALETTE[2], '-'),
             'two-mode PTT, unequal eps': (PALETTE[2], '--'), 'Oldroyd-B, eta_s/eta0 = 0.1': (PALETTE[3], '-'),
             'Rouse chain': (INK, '-.')}
    label = {'K-BKZ, one mode, a = 0.05': 'K-BKZ, one mode, $a=0.05$', 'K-BKZ, one mode, a = 0.2': 'K-BKZ, one mode, $a=0.2$',
             'two-mode PTT, common eps': r'two-mode PTT, common $\varepsilon$', 'two-mode PTT, unequal eps': r'two-mode PTT, unequal $\varepsilon$',
             'Oldroyd-B, eta_s/eta0 = 0.1': r'Oldroyd-B, $\eta_s/\eta_0=0.1$'}
    for key, (c, ls) in style.items():
        p = np.array(d['curves'][key], float)
        ax.semilogx(p[:, 0], p[:, 1], color=c, ls=ls, label=label.get(key, key))
    ax.axhline(1, color=MUTE, lw=0.6, ls=':')
    ax.set_xlim(1e-2, 20)
    ax.set_ylim(0, 2.75)
    ax.set_xlabel(r'$\dot\epsilon_{\rm eff}\bar\tau$')
    ax.set_ylabel(r'$\eta_{[P1]}/\eta_{P1}$')
    ax.legend(loc='upper left', fontsize=6.3, handlelength=2.2)
    fig.savefig(f'{FIGS}/fig4_outside.pdf')
    plt.close(fig)


def check(d):
    ck = Checks('fig4')
    e, c, t = d['WM_at_5']
    ck.add('White-Metzner at tau0 gdot = 5: composite / eta0', '1.13', c)
    ck.add('White-Metzner at the same eps_eff: eta_P1 / eta0', '2.23', t)
    k = np.array(d['curves']['K-BKZ, one mode, a = 0.05'])
    i = np.argmax(k[:, 1])
    ck.add('K-BKZ a = 0.05: largest overshoot of the composite (76%)', '1.76', k[i, 1])
    ck.add('K-BKZ a = 0.05: rate of the largest overshoot, eps_eff tau', '1.3', k[i, 0], tol=0.1)
    u = np.array(d['curves']['two-mode PTT, unequal eps'])
    ck.add('PTT, unequal eps: largest overshoot (59%)', '1.59', u[:, 1].max())
    cm = np.array(d['curves']['two-mode PTT, common eps'])
    ck.add('PTT, common eps: composite never exceeds the planar viscosity', '1.000', cm[:, 1].max())
    ck.add('PTT, common eps: high-rate Cauchy-Schwarz factor', '0.890', d['PTT_plateau_factor'])
    ck.add('PTT, common eps: numerical ratio at gdot taubar = 1e6', '0.890', d['PTT_plateau_numeric'], tol=0.002)
    if 'scan' in d:
        s = d['scan']
        ck.add('scan, eps >= 0.03: smallest ratio', '0.28', min(s['0.3'][0], s['0.1'][0], s['0.03'][0]))
        ck.add('scan, eps >= 0.03: largest ratio', '1.00', max(s['0.3'][1], s['0.1'][1], s['0.03'][1]))
        ck.add('scan, eps = 0.003: range low', '0.06', s['0.003'][0])
        ck.add('scan, eps = 0.003: range high', '1.06', s['0.003'][1])
        ck.add('scan, eps = 0.001: range low', '0.02', s['0.001'][0])
        ck.add('scan, eps = 0.001: range high', '1.17', s['0.001'][1])
    return ck.save()


if __name__ == '__main__':
    d = load_json('fig4.json') if '--plot' in sys.argv else compute(scan='--no-scan' not in sys.argv)
    plot(d)
    check(d)
