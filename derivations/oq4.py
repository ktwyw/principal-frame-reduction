import numpy as np
exec(open('oq2.py').read().split("print('(a) proof ingredients')")[0])
for al in (0.25, 0.5, 2/3):
    lv = np.linspace(-7, 7, 421); nk = 10**lv; Gk = nk**al; Hs = np.array([0.05, 0.1, 0.2]); Sk, Nk = protocol(Gk, nk, 1.0, np.r_[1e-3, Hs], t0=1e-9)
    D = np.hypot(Nk.sum(0), 2*Sk.sum(0))[1:]; r = D/planar(Gk, nk, 1.0, Hs); gam = None
    print(al, 2*al*(1-al)/((3-al)*(2-al)**3), (1 - r)/Hs**2)
# exponential-shear bound and strain at given Hencky strain for the power law alpha = 1/2
