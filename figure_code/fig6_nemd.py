"""
Figure 6 (and Table 4) -- test of the composite against the non-equilibrium molecular dynamics (NEMD) of
Daivis, Matin and Todd for melts of bead-rod chains (Sec. VII of the manuscript).

Data sources
 (i)  Third-order-fluid constants of J. Non-Newtonian Fluid Mech. 147, 35 (2007), Tables 2 and 3:
      eta0 = b1, Y = 2(b12 - b1:11) [eta(gdot) = b1 - Y gdot^2], Psi1,0 = -2 b2, and X1 = (b3 - 2 b12 + 2 b1:11) from
      eta_P1 = 4 b1 + 16 X1 eps^2 (X2: the same from eta_P2), with the standard errors quoted there.
 (ii) Curves digitised from J. Non-Newtonian Fluid Mech. 111, 1 (2003), Figs. 1, 2 and 4 (constant-volume data unless
      noted; constant normal pressure P_zz for eta1P). Strain rates lie on the simulation grid {1, 2, 5} x 10^n
      (or its square-root midpoints) and are exact; ordinates carry about 3% digitisation error (about 8% where markers
      overlap). The digitisation itself (marker extraction from the published bitmaps) is not repeated here.

Figure: eta_P1 from NEMD (circles) and the composite eta_[P1](eps_eff) built from the NEMD shear data (stars),
with Maxwell modes with the Rouse spectrum of an N-bead chain, tau_p ~ 1/sin^2(p pi/2N), scaled to taubar = Psi1,0/2eta0.

Run:  python3 fig6_nemd.py   (seconds)
"""
import numpy as np
from common import setup_matplotlib, panel_label, save_json, PALETTE, INK, MUTE, FIGS, Checks

# --------------------------------------------------------------------------------------------- (i) 2007 constants
# N: (eta0, d eta0, Y, dY, Psi10, dPsi10, X1, dX1, X2, dX2)
T2007 = {2: (2.957, .002, 4.9, .1, 3.44, .07, -2.6, .1, 0.30, .09),
         4: (4.307, .006, 53, 2, 20.5, .3, -5.9, .6, 6.6, .8),
         10: (8.42, .04, 3900, 340, 320, 10, 1100, 100, 880, 250),
         20: (15.6, .1, 105000, 9000, 3090, 240, 94000, 13000, 14000, 4000),
         50: (41.3, .6, 1.09e7, .09e7, 56000, 3000, 29e6, 3e6, 2.5e6, .9e6),
         100: (108, 3, 7e8, 1e8, 1.10e6, .04e6, 1.1e9, .9e9, None, None)}


def third_order_analysis():
    """b3 = X1 + Y; b2^2/b1 = Psi10^2/(4 eta0); defect coefficient b3 - b2^2/b1 (Eq. d1); fraction b2^2/(b1 b3);
    measured curvature X1 of eta_P1/16 against the curvature implied by the composite, b2^2/b1 - Y (Appendix A)."""
    rows = {}
    for N, (e0, de0, Y, dY, P, dP, X1, dX1, X2, dX2) in T2007.items():
        b3, db3 = X1 + Y, np.hypot(dX1, dY)
        q = P * P / (4 * e0)
        dq = q * np.hypot(2 * dP / P, de0 / e0)
        r = q / b3
        rows[N] = dict(taubar=P / (2 * e0), b3=b3, db3=db3, q=q, dq=dq, defect=b3 - q, ddefect=np.hypot(db3, dq),
                       fraction=r, dfraction=r * np.hypot(dq / q, db3 / b3), X1=X1, dX1=dX1,
                       comp_curv=q - Y, dcomp_curv=np.hypot(dq, dY))
        if X2 is not None:                       # same analysis with the eta_P2-based coefficient
            b3b = X2 + Y
            rows[N]['P2_based_sigmas'] = (b3b - q) / np.hypot(np.hypot(dX2, dY), dq)
    return rows


# --------------------------------------------------------------------------------------------- (ii) 2003 curves
a = lambda *v: np.array(v, float)
D50 = dict(N=50, eta0=41.3, Psi10=56000.0,
           gd=a(3.162e-4, 5e-4, 7.071e-4, 1e-3, 1.581e-3, 2.236e-3, 3.162e-3, 5e-3, 7.071e-3, 1e-2, 1.581e-2, 2.236e-2, 3.162e-2, 5e-2, 7.071e-2),
           eta=a(41.64, 36.77, 34.75, 29.36, 26.49, 23.2, 20.16, 16.76, 14.17, 12.88, 10.17, 8.57, 6.94, 6.07, 5.44),
           Psi1=a(6.553e4, 5.137e4, 4.83e4, 3.762e4, 2.654e4, 1.903e4, 1.31e4, 8051, 5091, 3496, 1801, 1253, 748, 430.9, 270.7),
           eps=a(1.118e-4, 1.581e-4, 2.5e-4, 3.536e-4, 5e-4, 7.906e-4, 1.118e-3, 1.581e-3, 2.5e-3, 3.536e-3, 5e-3, 7.906e-3, 1.118e-2, 1.581e-2, 2.5e-2, 3.536e-2, 5e-2, 7.906e-2),
           eta1V=a(180.9, 171.1, 206.7, 228.3, 282.8, 421, 489.7, 560.8, 594.9, 592.6, 572.3, 528.9, 540.8, 536.7, 520.6, 543.3, 626.4, 812.9),
           eta1P=a(180.9, 171.1, 206.7, 228.3, 282.8, 421, 489.7, 560.8, 594.9, 592.6, 572.3, 528.9, 472.0, 393.9, 309.1, 246.0, 187.7, 115.5))
D20 = dict(N=20, eta0=15.6, Psi10=3090.0,
           gd=a(2.236e-3, 3.162e-3, 5e-3, 7.071e-3, 1e-2, 1.581e-2, 2.236e-2, 3.162e-2, 5e-2, 7.071e-2),
           eta=a(14.56, 14.45, 13.05, 12.35, 11.24, 9.44, 8.50, 6.94, 6.07, 5.44),
           Psi1=a(2643, 2470, 2119, 1635, 1232, 790.5, 552.2, 364.5, 212.2, 137.7),
           eps=a(3.536e-4, 1.118e-3, 1.581e-3, 2.5e-3, 3.536e-3, 5e-3, 7.906e-3, 1.118e-2, 1.581e-2, 2.5e-2, 3.536e-2, 5e-2, 7.906e-2, 1.118e-1),
           eta1V=a(61.39, 66.36, 68.37, 72.62, 80.05, 95.32, 109.5, 117.4, 119, 117.5, 113.6, 111.9, 115.5, 112.7),
           eta1P=a(61.39, 66.36, 68.37, 72.62, 80.05, 95.32, 109.5, 117.4, 119, 109.8, 101.0, 87.76, 70.27, 57.39))


def rouse_maxwell(N, eta0, taubar, eps):
    """Maxwell modes with the Rouse spectrum of an N-bead chain, tau_p = c/sin^2(p pi/2N), p = 1..N-1, equal moduli,
    scaled so that the viscosity-weighted mean time equals taubar. Returns tau_1 and eta_P1(eps) from Eq. (multi)."""
    p = np.arange(1, N)
    t = 1 / np.sin(p * np.pi / (2 * N))**2
    t *= taubar / ((t * t).sum() / t.sum())
    G = eta0 / t.sum()
    return t[0], np.array([4 * G * (t / (1 - 4 * t * t * x * x)).sum() if 2 * t[0] * x < 1 else np.nan for x in eps])


def composite_curves(d):
    """Composite from the NEMD shear data: N1/2s12 = Psi1 gdot/(2 eta), eps_eff = gdot/(2 sqrt(1 + r^2)),
    eta_[P1] = 4 eta (1 + r^2); the NEMD eta_P1 is interpolated (log-log) to eps_eff for the ratio."""
    r = d['Psi1'] * d['gd'] / (2 * d['eta'])
    e = d['gd'] / (2 * np.sqrt(1 + r * r))
    c = 4 * d['eta'] * (1 + r * r)
    interp = lambda y: np.exp(np.interp(np.log(e), np.log(d['eps']), np.log(y)))
    taubar = d['Psi10'] / (2 * d['eta0'])
    tau1, _ = rouse_maxwell(d['N'], d['eta0'], taubar, [0.0])
    w = np.logspace(-4.2, np.log10(0.4999 / tau1), 80)
    _, rz = rouse_maxwell(d['N'], d['eta0'], taubar, w)
    return dict(r=r, eeff=e, comp=c, trueV=interp(d['eta1V']), trueP=interp(d['eta1P']), taubar=taubar, tau1=tau1, rouse_w=w, rouse=rz)


def plot(curves):
    import matplotlib.ticker
    plt = setup_matplotlib()
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
    for ax, d, tag, xl, yl in [(axs[0], D50, '(a) $N=50$', (8e-5, 0.12), (60, 2000)), (axs[1], D20, '(b) $N=20$', (2.5e-4, 0.16), (25, 700))]:
        c = curves[d['N']]
        ax.loglog(d['eps'], d['eta1V'], 'o', mfc=PALETTE[0], mec=PALETTE[0], ms=4, label=r'NEMD $\eta_{P1}$, constant volume')
        ax.loglog(d['eps'], d['eta1P'], 'o', mfc='none', mec=PALETTE[0], ms=4, mew=0.9, label=r'NEMD $\eta_{P1}$, constant $P_{zz}$')
        ax.loglog(c['eeff'], c['comp'], '*', mfc='none', mec=PALETTE[1], ms=7.5, mew=1.0, label=r'composite $\eta_{[P1]}$ from NEMD shear data')
        ax.loglog(c['rouse_w'], c['rouse'], color=INK, ls='--', lw=1.0, label='Maxwell modes with Rouse spectrum')
        ax.axhline(4 * d['eta0'], color=MUTE, ls=':', lw=0.7)
        ax.axvline(0.5 / c['tau1'], color=MUTE, ls=':', lw=0.7)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_xlabel(r'$\dot\epsilon$, $\dot\epsilon_{\rm eff}$  [reduced units]')
        ax.set_ylabel(r'$\eta_{P1}$  [reduced units]')
        panel_label(ax, tag)
        ax.text(xl[0] * 1.15, 4 * d['eta0'] * 0.86, r'$4\eta_0$', fontsize=7, color=MUTE, va='top')
        ax.text(0.5 / c['tau1'] * 1.08, yl[0] * 1.12, r'$1/2\tau_1$', fontsize=7, color=MUTE)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[1].legend(loc='upper right', fontsize=6.5)
    fig.savefig(f'{FIGS}/fig6_nemd.pdf')
    plt.close(fig)


def check(rows, curves):
    ck = Checks('fig6')
    for N, q in [(2, '0.43'), (4, '0.52'), (10, '0.61'), (20, '0.77'), (50, '0.48')]:
        ck.add(f'Table 4: b2^2/(b1 b3) for N = {N}', q, rows[N]['fraction'])
    ck.add('Table 4: b3 - b2^2/b1 > 0 for N = 2..50 (smallest value / its error)', '1', float(min(rows[N]['defect'] for N in (2, 4, 10, 20, 50)) > 0))
    ck.add('measured / composite curvature of eta_P1, N = 50 (3.6 times)', '3.6', rows[50]['X1'] / rows[50]['comp_curv'])
    ck.add('opposite sign for N = 10 (1 = yes)', '1', float(rows[10]['X1'] * rows[10]['comp_curv'] < 0))
    ck.add('eta_P2-based b3: b1 b3 < b2^2 for N = 20 by (standard errors)', '1.3', -rows[20]['P2_based_sigmas'], tol=0.06)
    ck.add('eta_P2-based b3: b1 b3 < b2^2 for N = 50 by (standard errors)', '2.3', -rows[50]['P2_based_sigmas'], tol=0.06)
    c50, c20 = curves[50], curves[20]
    ck.add('N = 50: tau_1/taubar', '1.52', c50['tau1'] / c50['taubar'])
    ck.add('N = 20: tau_1/taubar', '1.51', c20['tau1'] / c20['taubar'])
    ck.add('N = 50: Rouse coil-stretch rate 1/2tau_1', '4.9e-4', 0.5 / c50['tau1'])
    ck.add('N = 20: Rouse coil-stretch rate 1/2tau_1', '3.3e-3', 0.5 / c20['tau1'])
    ck.add('N = 50: maximum of eta_P1 / 4 eta0 (constant P_zz, first peak)', '3.6', D50['eta1P'].max() / (4 * D50['eta0']))
    ck.add('N = 20: maximum of eta_P1 / 4 eta0', '1.9', D20['eta1V'].max() / (4 * D20['eta0']))
    ck.add('N = 50: largest N1/2s12', '1.8', c50['r'].max())
    ck.add('N = 20: largest N1/2s12', '0.9', c20['r'].max())
    sel = c50['eeff'] * c50['taubar'] <= 1.05
    ck.at_most('N = 50: composite within 8% of 4 eta0 up to eps_eff taubar ~ 1', '0.08', np.max(abs(c50['comp'][sel] / (4 * D50['eta0']) - 1)))
    sel = c20['eeff'] * c20['taubar'] <= 0.42
    ck.at_most('N = 20: composite within 8% of 4 eta0 up to eps_eff taubar ~ 0.4', '0.08', np.max(abs(c20['comp'][sel] / (4 * D20['eta0']) - 1)))
    rr = c50['comp'] / c50['trueV']
    ck.add('N = 50: ratio at eps_eff taubar ~ 2', '0.25', np.interp(np.log(2.0), np.log(c50['eeff'] * c50['taubar']), rr), tol=0.03)
    ck.add('N = 50: ratio at the highest rate, constant-volume data', '0.17', rr[-1])
    ck.add('N = 50: ratio at the highest rate, constant-normal-pressure data', '0.24', (c50['comp'] / c50['trueP'])[-1])
    ck.add('abstract: factor four to six (1/ratio, constant volume)', '6', 1 / rr[-1], tol=0.5)
    ck.add('abstract: factor four to six (1/ratio, constant P_zz)', '4', 1 / (c50['comp'] / c50['trueP'])[-1], tol=0.5)
    ck.add('N = 20: smallest ratio', '0.34', (c20['comp'] / c20['trueV']).min())
    return ck.save()


if __name__ == '__main__':
    rows = third_order_analysis()
    curves = {d['N']: composite_curves(d) for d in (D50, D20)}
    save_json('fig6.json', dict(table4={str(k): v for k, v in rows.items()},
                                curves={str(k): {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv) for kk, vv in v.items()} for k, v in curves.items()}))
    plot(curves)
    check(rows, curves)
