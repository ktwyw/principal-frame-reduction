#!/bin/sh
# Re-run every derivation/check script; each writes its own .json next to it (see README.md). Total ~1 h on 2 cores.
for s in th2_2 th2_3 th2_4 th2_5 th2_7 oq_nearsf oq_fifth oq_powerlaw oq_rouse oq_bound oq1 oq2 oq3 oq4 oq5 check9 other_ext spectra; do
  echo "=== $s"; python3 -u $s.py > $s.log 2>&1 && echo ok || echo FAILED
done
