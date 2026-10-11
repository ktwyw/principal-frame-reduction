"""
Figure 5 -- FENE dumbbells with b = 50: closure versus no closure (Sec. VI of the manuscript).

Panel (a): steady shear viscosity and N1 from Brownian dynamics (symbols, data/fene_bd_records.json) and from the
           FENE-P closure (lines, full-tensor steady solver).
Panel (b): planar extensional viscosity: FENE-P (line; for FENE-P the composite and the planar viscosity coincide),
           true eta_P1 of FENE dumbbells from BD of planar extension (filled circles), and the composite eta_[P1]
           built from the BD shear data (stars). Each star is to be compared with the circle at the same abscissa,
           because each extension run was made at exactly the composite rate.

The Brownian dynamics take several hours on two cores. By default this script plots the stored results; to regenerate
them run   python3 fene_bd.py run data/fene_bd_records.json   or   python3 fig5_fene.py --recompute .

Run:  python3 fig5_fene.py   (about 1 min with stored BD data)
"""
import os
import sys
import json
import numpy as np
from common import (fenep, steady_state, L_shear, L_ext, composite_from_shear, setup_matplotlib, panel_label, save_json,
                    load_json, PALETTE, INK, FIGS, DATA, Checks)

BD_FILE = os.path.join(DATA, 'fene_bd_records.json')


def compute():
    if '--recompute' in sys.argv:
        import fene_bd
        from multiprocessing import Pool
        with Pool() as p:
            recs = p.map(fene_bd.case, fene_bd.CASES, chunksize=1)
        json.dump(recs, open(BD_FILE, 'w'), default=float)
    m = fenep(50.0)
    fp = {'shear': [], 'ext': []}
    for g in np.logspace(-1, 2.3, 34):
        s = m['stress'](steady_state(m, L_shear(g)))
        e, c1, _ = composite_from_shear(s, g)
        fp['shear'].append([g, s[0, 1] / g, s[0, 0] - s[1, 1], e, c1])
    for e in np.logspace(-1.3, 1, 60):
        s = m['stress'](steady_state(m, L_ext(e), T=4000))
        fp['ext'].append([e, (s[0, 0] - s[1, 1]) / e, (s[2, 2] - s[1, 1]) / e])
    # FENE-P closure error against the BD values at the same Wi (text: N1 overestimated by 31% at Wi = 3, 53% at Wi = 100)
    recs = json.load(open(BD_FILE))
    over = {}
    for r in recs:
        if r['b'] == 50.0 and r['tag'] == 'curve' and r['shear']['g'] in (3.0, 100.0):
            g = r['shear']['g']
            s = m['stress'](steady_state(m, L_shear(g)))
            over[str(g)] = (s[0, 0] - s[1, 1]) / r['shear']['N1'] - 1
    # FENE-P vs FENE planar viscosity at the BD extension rates
    ext_cmp = []
    for r in recs:
        if r['b'] == 50.0 and r['tag'] == 'curve':
            e = r['ext']['e']
            s = m['stress'](steady_state(m, L_ext(e), T=4000))
            ext_cmp.append([e, ((s[0, 0] - s[1, 1]) / e) / r['ext']['P1'] - 1])
    out = dict(fenep=fp, N1_overestimate=over, ext_fenep_vs_fene=sorted(ext_cmp))
    save_json('fig5.json', out)
    return out


def plot(d):
    import matplotlib.ticker
    plt = setup_matplotlib()
    recs = [r for r in json.load(open(BD_FILE)) if r['b'] == 50.0 and r['tag'] == 'curve']
    recs.sort(key=lambda r: r['shear']['g'])
    sh, ex = np.array(d['fenep']['shear']), np.array(d['fenep']['ext'])
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.75), constrained_layout=True)
    ax = axs[0]
    g = np.array([r['shear']['g'] for r in recs])
    ax.loglog(sh[:, 0], sh[:, 1], color=PALETTE[0], label=r'$\eta$, FENE-P')
    ax.loglog(g, [r['shear']['eta'] for r in recs], 'o', mfc='none', mec=PALETTE[0], ms=4.5, label=r'$\eta$, FENE (BD)')
    ax.loglog(sh[:, 0], sh[:, 2], color=PALETTE[1], ls='--', label=r'$N_1$, FENE-P')
    ax.loglog(g, [r['shear']['N1'] for r in recs], 's', mfc='none', mec=PALETTE[1], ms=4.5, label=r'$N_1$, FENE (BD)')
    ax.set_xlabel(r'$\dot\gamma\lambda_H$')
    ax.set_ylabel(r'$\eta/nkT\lambda_H$,  $N_1/nkT$')
    ax.legend(loc='upper left')
    panel_label(ax, '(a)', 'lower right')
    ax = axs[1]
    ax.loglog(ex[:, 0], ex[:, 1], color=INK, lw=1.1, label=r'FENE-P: $\eta_{P1}=\eta_{[P1]}$')
    ax.loglog([r['ext']['e'] for r in recs], [r['ext']['P1'] for r in recs], 'o', mfc=PALETTE[0], mec=PALETTE[0], ms=4.5,
              label=r'FENE (BD): true $\eta_{P1}$')
    ax.loglog([r['shear']['eps'] for r in recs], [r['shear']['P1'] for r in recs], '*', mfc='none', mec=PALETTE[1], ms=7.5,
              mew=1.0, label=r'FENE (BD): composite $\eta_{[P1]}$')
    ax.set_xlabel(r'$\dot\epsilon\lambda_H$,  $\dot\epsilon_{\rm eff}\lambda_H$')
    ax.set_ylabel(r'$\eta_{P1}/nkT\lambda_H$')
    ax.legend(loc='upper left')
    ax.set_xlim(0.1, 10)
    ax.set_ylim(3, 200)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    panel_label(ax, '(b)', 'lower right')
    fig.savefig(f'{FIGS}/fig5_fene.pdf')
    plt.close(fig)


def check(d):
    ck = Checks('fig5')
    recs = json.load(open(BD_FILE))
    ratio = {(r['tag'], r['b'], r['shear']['g']): r['ratio1'] for r in recs}
    for g, q in [(2.0, '0.87'), (3.0, '0.70'), (5.0, '0.52')]:
        ck.add(f'BD: composite/true at Wi = {g:g}', q, ratio[('curve', 50.0, g)])
    hi = [ratio[('curve', 50.0, g)] for g in (10.0, 20.0, 30.0, 50.0, 100.0)]
    ck.add('BD: smallest ratio for Wi >= 10 (43%)', '0.43', min(hi))
    ck.add('BD: largest ratio for Wi >= 10 (47%)', '0.47', max(hi))
    for g in (0.5, 1.0):
        ck.add(f'BD: ratio at Wi = {g:g} within 1%', '1.00', ratio[('curve', 50.0, g)], tol=0.01)
    ck.add('BD: Wi = 10, halved time step', '0.467', ratio[('conv-dt', 50.0, 10.0)], tol=0.0015)
    ck.add('BD: Wi = 10, other seed', '0.467', ratio[('conv-seed', 50.0, 10.0)], tol=0.0015)
    ck.add('BD: b = 20', '0.48', ratio[('b', 20.0, 10.0)])
    ck.add('BD: b = 100', '0.46', ratio[('b', 100.0, 10.0)])
    ck.add('FENE-P overestimates N1 at Wi = 3 (31%)', '0.31', d['N1_overestimate']['3.0'])
    ck.add('FENE-P overestimates N1 at Wi = 100 (53%)', '0.53', d['N1_overestimate']['100.0'])
    e = np.array(d['ext_fenep_vs_fene'])
    ck.add('FENE-P vs FENE planar viscosity for eps lambda >= 1.3: below 5% (|difference| - 0.025 within 0.025)', '0.025', max(abs(e[e[:, 0] >= 1.29, 1])), tol=0.025)
    ck.add('FENE-P vs FENE planar viscosity: up to 60% near the transition', '0.6', max(abs(e[:, 1])), tol=0.05)
    return ck.save()


if __name__ == '__main__':
    d = load_json('fig5.json') if '--plot' in sys.argv else compute()
    plot(d)
    check(d)
