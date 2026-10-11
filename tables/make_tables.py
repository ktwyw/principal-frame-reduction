"""
make_tables.py -- regenerate Tables 2-6 of the manuscript from data, so that no number is transcribed by hand.
(Table 1 is produced by class_table.py.)  Output: tables/output/*.tex, compared with tables/manuscript/*.tex.

  Table 2 (converse_table.tex)  from converse_3d.json  (python3 converse_3d.py, ~2 min)
  Table 3 (flows_table.tex)     computed here: steady states of four models in two 3D flows (~2 min)
  Table 4 (nemd_table.tex)      from the constants of Daivis, Matin & Todd (2007), via ../figure_code/fig6_nemd.py
  Table 5 (bd_table.tex)        from ../figure_code/data/fene_bd_records.json (Brownian dynamics, see ../figure_code/fene_bd.py)
  Table 6 (coax_table.tex)      from fene_bd3.json (python3 fene_bd3.py, Brownian dynamics, ~30 min)

Run:  python3 make_tables.py            (add --skip-3d to reuse output/flows_table.tex)
Numbers at round-off level (mismatches of order 1e-16) differ from run to run and are not compared.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'figure_code'))
from captions import CAPTIONS                      # noqa: E402

OUT = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)


def sci(v):
    """a x 10^b with one significant digit, for round-off-level mismatches."""
    e = int(np.floor(np.log10(v)))
    return f"${v / 10**e:.0f}\\times10^{{{e}}}$"


def wrap(name, colspec, header, rows, star=False):
    env = 'table*' if star else 'table'
    return (f"\\begin{{{env}}}\n\\caption{{{CAPTIONS[name]}}}\n" + ("\\footnotesize\n" if name == 'nemd_table' else "")
            + f"\\begin{{ruledtabular}}\n\\begin{{tabular}}{{{colspec}}}\n{header}\\\\\n\\hline\n" + "\n".join(rows)
            + "\n\\end{tabular}\n\\end{ruledtabular}\n" + f"\\end{{{env}}}\n")


# ------------------------------------------------------------------ Table 2
def table2():
    d = json.load(open(os.path.join(HERE, 'theory2_6.json')))
    label = {'spin': r'Modified spin $\bm W+\kappa(\bm A\!\cdot\!\bm D-\bm D\!\cdot\!\bm A)$',
             '[\\bm A,[\\bm A,\\bm D^2]]': r'$[\bm A,[\bm A,\bm D^2]]$',
             '(\\tr\\bm D^2)': r'$(\tr\bm D^2)\,[\bm A,[\bm A,\bm D]]$',
             'D_{\\!A}^2$ (quadratic': r'$-\bm D_{\!A}^2$ (quadratic in $\bm D$)',
             'relaxation rate $1+\\kappa\\tr\\bm D_{\\!A}^2$': r'Relaxation rate $1+\kappa\tr\bm D_{\!A}^2$',
             '\\bm A\\!\\cdot\\!\\bm D^2+': r'$\bm A\!\cdot\!\bm D^2+\bm D^2\!\cdot\!\bm A$',
             'relaxation rate $1+\\kappa\\tr\\bm D^2$': r'Relaxation rate $1+\kappa\tr\bm D^2$'}

    def name(k):
        for key, v in label.items():
            if key in k:
                return v
        raise KeyError(k)
    fmt = lambda v: sci(v) if v < 1e-3 else f"{100 * v:.0f}\\%"
    inside = [f"{name(k)} & {fmt(v[0])}\\\\" for k, v in d.items() if v[1]]
    outside = [f"{name(k)} & {fmt(v[0])}\\\\" for k, v in d.items() if not v[1]]
    rows = (["\\multicolumn{2}{l}{\\emph{Diagonal part independent of $\\bm D_{\\!\\perp}$}}\\\\"] + inside + ["\\hline",
            "\\multicolumn{2}{l}{\\emph{Diagonal part dependent on $\\bm D_{\\!\\perp}$}}\\\\"] + outside)
    return wrap('converse_table', 'lc', 'Added term & Mismatch', rows)


# ------------------------------------------------------------------ Table 3
def table3():
    import lib
    flows = {'I': np.array([[-.15, 2., 0], [0, -.15, 0], [0, 0, .3]]), 'II': np.array([[.2, 1.5, .7], [0, -.3, .9], [.4, 0, .1]])}
    rows = []
    for mname, m in [('linear PTT', lib.lptt()), ('Giesekus', lib.giesekus(0.2)), ('Rolie-Poly', lib.rolie_poly()), ('FENE-P', lib.fenep())]:
        for fname, L in flows.items():
            A = lib.steady(m, L)
            D = (L + L.T) / 2
            w, V = np.linalg.eigh(A)
            d = np.array([V[:, i] @ D @ V[:, i] for i in range(3)])
            B = lib.steady(m, np.diag(d), T=1e4)                     # shear-free flow, computed independently from rest
            sa = np.linalg.eigvalsh(m['stress'](A))
            sb = np.sort(np.diag(m['stress'](B)))
            err = float(np.max(np.abs(sa - sb)) / np.max(np.abs(sa)))
            dd = sorted(d, reverse=True)
            rows.append(f"{mname} & {fname} & $({dd[0]:.2f},{dd[1]:.2f},{dd[2]:.2f})$ & {sci(max(err, 1e-17))} \\\\")
            print(f'  Table 3: {mname:10s} flow {fname:2s} d = {np.round(dd, 3)}, mismatch {err:.1e}', flush=True)
    return wrap('flows_table', 'llcc', r'Model & Flow & $(d_1,d_2,d_3)\tau$ & Mismatch', rows)


# ------------------------------------------------------------------ Table 4
def table4():
    from fig6_nemd import third_order_analysis
    R = third_order_analysis()

    def pm(v, e):
        ex = int(np.floor(np.log10(abs(v)))) if v != 0 else 0
        if abs(ex) < 3:
            dec = max(0, 1 - int(np.floor(np.log10(e))))
            return f"${v:.{dec}f}\\pm{e:.{dec}f}$"
        return f"$({v / 10**ex:.2f}\\pm{e / 10**ex:.2f})\\times10^{{{ex}}}$"
    rows = []
    for N in [2, 4, 10, 20, 50, 100]:
        r = R[N]
        tb = f"{r['taubar']:.0f}" if r['taubar'] >= 100 else f"{r['taubar']:.3g}"
        rows.append(f"{N} & {tb} & {pm(r['b3'], r['db3'])} & {pm(r['q'], r['dq'])} & ${r['fraction']:.2f}\\pm{r['dfraction']:.2f}$ & "
                    f"{pm(r['X1'], r['dX1'])} & {pm(r['comp_curv'], r['dcomp_curv'])}\\\\")
    hdr = r'$N$ & $\bar\tau$ & $b_3$ & $b_2^2/b_1$ & $b_2^2/b_1b_3$ & $\eta_{P1}$ curvature, NEMD & composite'
    return wrap('nemd_table', 'ccccccc', hdr, rows, star=True)


# ------------------------------------------------------------------ Tables 5 and 6
def table5():
    recs = json.load(open(os.path.join(HERE, '..', 'figure_code', 'data', 'fene_bd_records.json')))

    def row(r, note=''):
        s, x = r['shear'], r['ext']
        return (f"{r['b']:.0f} & {s['g']:g} & {s['eta']:.3f} & {s['N1']:.2f} & {s['eps']:.3f} & {s['P1']:.2f} & "
                f"{x['P1']:.2f} & {r['ratio1']:.3f} & {note}\\\\")
    main = sorted([r for r in recs if r['b'] == 50.0 and r['tag'] == 'curve'], key=lambda r: r['shear']['g'])
    extra = [(r, {'conv-dt': 'half time step', 'conv-seed': 'new seed', 'b': ''}[r['tag']]) for r in recs if r['tag'] in ('conv-dt', 'conv-seed', 'b')]
    extra.sort(key=lambda t: (t[0]['b'] != 50.0, t[0]['b'], t[1]))
    rows = [row(r) for r in main] + ["\\hline"] + [row(r, n) for r, n in extra]
    hdr = r'$b$ & Wi & $\eta$ & $N_1$ & $\eeff$ & $\eta_{[P1]}$ & $\eta_{P1}$ (true) & $\eta_{[P1]}/\eta_{P1}$ & '
    return wrap('bd_table', 'ccccccccc', hdr, rows, star=True)


def table6():
    B = sorted(json.load(open(os.path.join(HERE, 'fene_bd3.json'))), key=lambda r: r['g'])
    rows = [f"{r['g']:g} & {r['d'][0]:.2f} & {r['d'][1]:.2f} & {r['d'][2] / r['d'][4]:.2f} & {r['d'][3] / r['d'][5]:.2f} & "
            f"{r['d'][6]:.3f} & {r['d'][7]:.3f} \\\\" for r in B]
    hdr = r'Wi & $\chi_\sigma$ & $\chi_C$ & $\sigma_{12}^{\rm P}/\sigma_{12}$ & $N_1^{\rm P}/N_1$ & $\langle Q^2\rangle/b$ & $P_{Q^2>b/2}$'
    return wrap('coax_table', 'ccccccc', hdr, rows)


def body(tex):
    """Table rows with round-off-level numbers (x10^-15 or smaller) masked, for comparison."""
    t = tex[tex.index('\\hline'):tex.index('\\end{tabular}')]
    return re.sub(r'\$\d\\times10\^\{-1[5-9]\}\$', '<roundoff>', t)


if __name__ == '__main__':
    jobs = [('converse_table', table2), ('flows_table', table3), ('nemd_table', table4), ('bd_table', table5), ('coax_table', table6)]
    ok_all = True
    for name, fn in jobs:
        path = os.path.join(OUT, name + '.tex')
        if name == 'flows_table' and '--skip-3d' in sys.argv and os.path.exists(path):
            tex = open(path).read()
        else:
            tex = fn()
            open(path, 'w').write(tex)
        ref = open(os.path.join(HERE, 'manuscript', name + '.tex')).read()
        same = body(tex) == body(ref)
        ok_all &= same
        print(f'{name:15s}: {"identical to the manuscript table" if same else "DIFFERS from the manuscript table"}')
        if not same:
            import difflib
            print('\n'.join(difflib.unified_diff(body(ref).split('\n'), body(tex).split('\n'), 'manuscript', 'generated', lineterm='', n=0)))
    print('all tables reproduced' if ok_all else 'some tables differ')
