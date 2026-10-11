Digitized molecular-dynamics data (Daivis, Matin & Todd, J. Non-Newtonian Fluid Mech. 111, 1 (2003), Figs. 1, 2 and 4).

* nemd_digitize.py: extracts marker positions from the bitmap figures of the paper (the figures themselves are not
  distributed; extract them from the published PDF with `pdfimages`).
* nemd_digitized_raw.json: its raw output.
* The curated series used in the paper (markers assigned to data sets, strain rates snapped to the simulation grid
  {1, 2, 5} x 10^n and its square-root midpoints) are typed into figure_code/fig6_nemd.py.
  Ordinates carry about 3 % digitization error (about 8 % where markers overlap); strain rates are exact.
* The third-order-fluid constants (2007 paper, Tables 2-3) are typed into figure_code/fig6_nemd.py with their
  quoted standard errors.
