"""
Figure 3 -- spectra of relaxation times and the slow-flow defect (Sec. V of the manuscript).

Panel (a): Rouse (tau_p = tau_1 p^-2) and Zimm (tau_p = tau_1 p^-3/2) chains with equal moduli, i.e. sums of Maxwell
           (UCM) modes. Exact planar viscosity, Eq. (multi):  eta_P1 = sum_k 4 eta_k/(1 - 4 tau_k^2 eps^2);
           composite:  eta_[P1] = 4 eta0/(1 - 4 taubar^2 eps^2),  taubar = sum eta_k tau_k / eta0.
Panel (b): (eta_P1 - eta_[P1]) / (16 eta0 Var(tau) eps^2) for four nonlinear fluids sharing the two-mode spectrum
           (eta0/2, 0.2 taubar), (eta0/2, 1.8 taubar): linear PTT and exponential PTT (eps = 0.1), Giesekus (alpha = 0.2),
           and a K-BKZ fluid with PSM damping (a = 0.2); plus the Rouse chain. Eq. (defect) predicts the limit 1.

Run:  python3 fig3_spectra.py   (about 2 min)
"""
import sys
import numpy as np
from scipy.special import zeta
from common import (lptt_shear, lptt_ext, eptt_shear, eptt_ext, giesekus_shear, giesekus_ext, multimode, kbkz_psm,
                    lptt, eptt, giesekus, steady_state, L_shear, L_ext, composite_from_shear,
                    setup_matplotlib, panel_label, save_json, load_json, PALETTE, INK, MUTE, FIGS, TWO_MODE, TWO_MODE_VAR, Checks)

P = 200000                       # number of chain modes (converged to plotting accuracy)


def chain(k, w):
    """Chain with tau_p = p^-k (units tau_1), equal moduli (eta_p proportional to tau_p): exact eta_P1/eta0 at
    eps tau_1 = w, and the composite."""
    t = np.arange(1, P + 1.0)**-k
    eta0 = t.sum()
    tbar = (t * t).sum() / eta0
    exact = 4 * (t / (1 - 4 * t * t * w * w)).sum() / eta0
    comp = 4 / (1 - 4 * tbar**2 * w * w) if 2 * tbar * w < 1 else None
    return exact, comp


def compute():
    out = {'chains': {}, 'defect': {}}
    for name, k in [('Rouse', 2.0), ('Zimm', 1.5)]:
        tbar, t2 = zeta(2 * k) / zeta(k), zeta(3 * k) / zeta(k)          # <tau>, <tau^2> in units of tau_1
        w = np.concatenate([np.linspace(0.005, 0.49, 98), [0.495, 0.498]])
        out['chains'][name] = dict(k=k, tbar=tbar, t2=t2, curve=[[x, *chain(k, x)] for x in w],
                                   comp=[[x, 4 / (1 - 4 * tbar**2 * x * x)] for x in np.linspace(0.005, 0.999 / (2 * tbar), 200)])
    modes = lambda p: [(h, t, p) for h, t in TWO_MODE]
    var = TWO_MODE_VAR
    for name, sh, ex, p in [('linear PTT', lptt_shear, lptt_ext, 0.1), ('Giesekus', giesekus_shear, giesekus_ext, 0.2),
                            ('exponential PTT', eptt_shear, eptt_ext, 0.1)]:
        pts = []
        for g in np.logspace(-1.7, -0.35, 14):
            e, c, tr = multimode(modes(p), g, sh, ex)
            pts.append([e, (tr - c) / (16 * var * e * e)])
        out['defect'][name] = pts
    out['defect']['K-BKZ (PSM, a = 0.2)'] = [[e, (tr - c) / (16 * var * e * e)]
                                            for e, c, tr in [kbkz_psm(TWO_MODE, 0.2, g) for g in np.logspace(-1.7, -0.5, 10)]]
    # second composite at eps taubar ~ 0.01 (text: 0.94-0.95), full-tensor solver, Eq. (d2)
    p2 = {}
    for name, make in [('linear PTT', lambda: lptt(0.1)), ('exponential PTT', lambda: eptt(0.1)), ('Giesekus', lambda: giesekus(0.2))]:
        m = make()
        g = 0.02
        s = sum(h / t * m['stress'](steady_state(m, L_shear(g * t), T=200)) for h, t in TWO_MODE)
        e, c1, c2 = composite_from_shear(s, g)
        x = sum(h / t * m['stress'](steady_state(m, L_ext(e * t), T=2000)) for h, t in TWO_MODE)
        p2[name] = [e, ((x[2, 2] - x[1, 1]) / e - c2) / (8 * var * e * e)]
    out['P2_check'] = p2
    save_json('fig3.json', out)
    return out


def plot(d):
    plt = setup_matplotlib()
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.75), constrained_layout=True)
    ax = axs[0]
    for (nm, ch), c in zip(d['chains'].items(), PALETTE):
        t = np.array([[r[0], r[1]] for r in ch['curve']])
        cp = np.array(ch['comp'])
        ax.semilogy(t[:, 0], t[:, 1], color=c, label=f'{nm}, exact')
        ax.semilogy(cp[:, 0], cp[:, 1], color=c, ls='--', label=f'{nm}, composite')
    ax.axvline(0.5, color=MUTE, lw=0.6, ls=':')
    ax.set_xlim(0, 1.15)
    ax.set_ylim(3, 300)
    ax.set_xlabel(r'$\dot\epsilon\tau_1$')
    ax.set_ylabel(r'$\eta_{P1}/\eta_0$')
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.87), fontsize=6.5)
    panel_label(ax, '(a)')
    ax = axs[1]
    for (nm, pts), c, mk in zip(d['defect'].items(), PALETTE, ['o', 's', '^', 'D']):
        p = np.array(pts)
        ax.semilogx(p[:, 0], p[:, 1], mk + '-', color=c, ms=3.8, mfc='none', mew=0.9, lw=0.8, label=nm)
    ch = d['chains']['Rouse']
    var = ch['t2'] - ch['tbar']**2
    r = np.array([[q[0], (q[1] - q[2]) / (16 * var * q[0]**2)] for q in ch['curve'] if q[2] and q[0] < 0.3])
    ax.semilogx(r[:, 0] * ch['tbar'], r[:, 1], color=INK, lw=1.0, ls='-.', label='Rouse chain (exact)')
    ax.axhline(1, color=MUTE, lw=0.6, ls=':')
    ax.set_ylim(0.3, 1.6)
    ax.set_xlim(8e-3, 0.3)
    ax.set_xlabel(r'$\dot\epsilon\bar\tau$')
    ax.set_ylabel(r'$(\eta_{P1}-\eta_{[P1]})\,/\,16\eta_0\mathrm{Var}(\tau)\dot\epsilon^2$')
    ax.legend(loc='upper left', title='two-mode fluids', title_fontsize=7)
    panel_label(ax, '(b)', 'lower right')
    fig.savefig(f'{FIGS}/fig3_spectra.pdf')
    plt.close(fig)


def check(d):
    ck = Checks('fig3')
    R, Z = d['chains']['Rouse'], d['chains']['Zimm']
    ck.add('Rouse <tau>/tau_1 = pi^2/15', '0.658', R['tbar'])
    ck.add('Rouse <tau>^2/<tau^2>', '0.70', R['tbar']**2 / R['t2'])
    ck.add('Zimm <tau>^2/<tau^2>', '0.524', Z['tbar']**2 / Z['t2'])
    ck.add('Rouse: composite diverges at eps tau_1', '0.76', 0.5 / R['tbar'])
    ck.add('Zimm: composite diverges at eps tau_1', '1.09', 0.5 / Z['tbar'])
    ex, co = chain(2.0, 0.4)
    ck.add('Rouse eta_P1/eta0 at eps tau_1 = 0.4', '8.35', ex)
    ck.add('Rouse eta_[P1]/eta0 at eps tau_1 = 0.4', '5.53', co)
    for nm, pts in d['defect'].items():
        ck.add(f'{nm}: scaled defect tends to 1 at the lowest rate', '1.0', pts[0][1], tol=0.03)
    for nm, (e, v) in d['P2_check'].items():
        ck.add(f'{nm}: scaled P2 defect at eps taubar = {e:.4f}', '0.945', v, tol=0.0075)
    ck.add('two-mode spectrum: eta_P1 at eps taubar = 0.25, single mode', '5.33', 4 / (1 - 4 * 0.0625))
    ck.add('two-mode spectrum: eta_P1 at eps taubar = 0.25, two modes', '12.55', sum(4 * h / (1 - 4 * t * t * 0.0625) for h, t in TWO_MODE))
    return ck.save()


if __name__ == '__main__':
    d = load_json('fig3.json') if '--plot' in sys.argv else compute()
    plot(d)
    check(d)
