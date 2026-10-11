"""
run_all.py -- regenerate every figure of the manuscript and verify every number quoted from it.

    python3 run_all.py /path/to/PIB_freqsweep_flowcurve.xlsx          compute everything (about 20 min on 2 cores)
    python3 run_all.py --plot                                          re-plot from the stored data in data/

Writes figures/fig1_collapse.pdf ... figures/fig7_pib.pdf, the data behind each figure in data/*.json, and
CHECKS.md, a table of every number quoted in the manuscript, the value recomputed here and the verdict.
The Brownian dynamics behind Fig. 5 are not rerun (several hours); see fene_bd.py.
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ['fig1_collapse.py', 'fig2_protocol.py', 'fig3_spectra.py', 'fig4_outside.py', 'fig5_fene.py', 'fig6_nemd.py', 'fig7_pib.py']


def main():
    plot_only = '--plot' in sys.argv
    xlsx = next((a for a in sys.argv[1:] if a.endswith('.xlsx')), None)
    subprocess.run([sys.executable, os.path.join(HERE, 'common.py')], check=True)        # self-test of the model library
    for s in SCRIPTS:
        args = [sys.executable, '-I', os.path.join(HERE, s)] if s == 'fig7_pib.py' else [sys.executable, os.path.join(HERE, s)]
        if plot_only:
            args.append('--plot')
        elif s == 'fig7_pib.py':
            if xlsx is None:
                print('fig7_pib.py skipped: give the path of PIB_freqsweep_flowcurve.xlsx')
                continue
            args.append(xlsx)
        print(f'\n===== {s}', flush=True)
        subprocess.run(args, check=True, cwd=HERE)
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'data', 'checks_fig*.json'))):
        rows += json.load(open(f))
    n_ok = sum(r['ok'] for r in rows)
    with open(os.path.join(HERE, 'CHECKS.md'), 'w') as f:
        f.write(f'# Numbers quoted in the manuscript, recomputed\n\n{n_ok} of {len(rows)} checks pass.\n\n')
        f.write('Default tolerance: half a unit in the last printed digit; "<=" marks upper-bound statements.\n\n')
        f.write('| Figure | Quantity | Manuscript | Recomputed | Verdict |\n|---|---|---|---|---|\n')
        for r in rows:
            f.write(f"| {r['figure']} | {r['what']} | {r['quoted']} | {r['computed']:.5g} | {'OK' if r['ok'] else '**FAIL**'} |\n")
    print(f'\n{n_ok} of {len(rows)} checks pass; see CHECKS.md')


if __name__ == '__main__':
    main()
