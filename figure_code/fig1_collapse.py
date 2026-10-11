"""
Figure 1 -- principal-frame reduction for planar flows (Sec. III D of the manuscript).

For the Giesekus (alpha_G = 0.2) and Rolie-Poly models we solve the full six-component tensor equations in steady
planar flows L = gdot (e1 e2 + alpha e2 e1) for alpha = 0 (simple shear), 0.3, -0.3 and 1 (planar extension),
project the stress onto its own principal axes, and plot
    (s+ - s-)/d1   and   (s3 - s-)/d1     against   d1 = n+.D.n+ .
The lines are the first and second planar extensional viscosities computed independently from rest in planar
extension. The theorem predicts exact collapse; the script reports the largest mismatch.

Run:  python3 fig1_collapse.py            (about 5-10 min; writes data/fig1.json and figures/fig1_collapse.pdf)
      python3 fig1_collapse.py --plot     (re-plot from data/fig1.json)
"""
import sys
import numpy as np
from common import (giesekus, rolie_poly, steady_state, L_ext, L_planar_family, planar_projection, planar_viscosities,
                    setup_matplotlib, panel_label, save_json, load_json, PALETTE, INK, FIGS, Checks)

MODELS = {'Giesekus': lambda: giesekus(0.2), 'Rolie-Poly': lambda: rolie_poly()}
ALPHAS = [0.0, 0.3, -0.3, 1.0]


def compute():
    out = {}
    for name, make in MODELS.items():
        m = make()
        rec = {'planar': [], 'flows': {}}
        # independent reference: steady planar extension from rest
        for e in np.logspace(-2, 1.3, 50):
            s = m['stress'](steady_state(m, L_ext(e), T=2000 if e < 0.2 else 4000))
            rec['planar'].append([e, *planar_viscosities(s, e)])
        # planar flows of different type, projected onto the stress eigenframe
        for al in ALPHAS:
            pts = []
            for g in np.logspace(-1.3, 2.5, 16):
                if al == 1.0 and g > 20:
                    continue
                L = L_planar_family(g, al)
                s = m['stress'](steady_state(m, L))
                pts.append([g, *planar_projection(s, L)])
            rec['flows'][str(al)] = pts
        # mismatch: compare each projected point with planar extension computed at exactly d1
        worst = 0.0
        for al in ALPHAS:
            for g, d1, p1, p2 in rec['flows'][str(al)]:
                x = m['stress'](steady_state(m, L_ext(d1), T=4000))
                t1, t2 = planar_viscosities(x, d1)
                worst = max(worst, abs(p1 / t1 - 1), abs(p2 / t2 - 1))
        rec['max_mismatch'] = worst
        print(f'{name}: largest relative mismatch between projected and true planar viscosities = {worst:.1e}')
        out[name] = rec
    save_json('fig1.json', out)
    return out


def plot(d):
    plt = setup_matplotlib()
    names = {'0.0': r'simple shear ($\alpha=0$)', '0.3': r'mixed, $\alpha=0.3$', '-0.3': r'elliptic, $\alpha=-0.3$',
             '1.0': r'$\alpha=1$ (pure extension)'}
    marker = {'0.0': 'o', '0.3': 's', '-0.3': '^', '1.0': 'D'}
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.75), constrained_layout=True)
    for ax, (mname, rec), tag in zip(axs, d.items(), ['(a)', '(b)']):
        t = np.array(rec['planar'])
        ax.loglog(t[:, 0], t[:, 1], color=INK, lw=1.1, label=r'planar extension, $\eta_{P1}$')
        ax.loglog(t[:, 0], t[:, 2], color=INK, lw=1.1, ls='--', label=r'planar extension, $\eta_{P2}$')
        for (al, pts), c in zip(rec['flows'].items(), PALETTE):
            p = np.array(pts)
            ax.loglog(p[:, 1], p[:, 2], marker[al], ms=4.2, mfc='none', mec=c, mew=1.0, label=names[al])
            ax.loglog(p[:, 1], p[:, 3], marker[al], ms=4.2, mfc='none', mec=c, mew=1.0)
        ax.set_xlabel(r'$d_1\tau$  (true rate $\dot\epsilon\tau$ for the lines)')
        ax.set_ylabel(r'$(\sigma_+-\sigma_-)/d_1\eta_0$,  $(\sigma_3-\sigma_-)/d_1\eta_0$')
        ax.set_xlim(8e-3, 30)
        panel_label(ax, f'{tag} {mname}')
    axs[0].legend(loc='lower left', handlelength=1.8)
    fig.savefig(f'{FIGS}/fig1_collapse.pdf')
    plt.close(fig)


def check(d):
    ck = Checks('fig1')
    for name, rec in d.items():
        ck.add(f'{name}: collapse is exact (mismatch < 1e-9)', '0', rec['max_mismatch'], tol=1e-9)
    return ck.save()


if __name__ == '__main__':
    d = load_json('fig1.json') if '--plot' in sys.argv else compute()
    plot(d)
    check(d)
