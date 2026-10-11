import numpy as np
from rlib import *
rows=[(50,0.5,0.900,0.39,0.229,4.29,4.28,1.002),(50,1,0.869,1.44,0.385,5.85,5.90,0.992),(50,2,0.791,4.63,0.564,9.95,11.40,0.873),(50,3,0.718,8.40,0.685,13.79,19.67,0.701),(50,5,0.608,16.13,0.881,19.56,37.44,0.522),(50,10,0.452,33.91,1.288,27.25,58.22,0.468),(50,20,0.318,63.63,1.958,33.14,72.74,0.456),(50,30,0.254,88.72,2.535,35.50,78.98,0.449),(50,50,0.188,131.47,3.544,37.47,84.99,0.441),(50,100,0.124,217.92,5.643,38.87,90.59,0.429),(50,10,0.452,33.89,1.289,27.22,58.28,0.467),(50,10,0.453,33.92,1.289,27.23,58.29,0.467),(20,10,0.351,19.77,1.674,12.53,25.91,0.484),(100,10,0.536,48.98,1.068,46.94,102.65,0.457)]
print('b Wi | eeff(tab) eeff(recomp) | [P1](tab) recomp | ratio tab recomp | chi_sigma | FENE-P: eta N1 N1P/N1bd  etaP/etabd | FENE-P planar at eeff / BD planar')
for b,Wi,eta,N1,ee,c,t,r in rows:
    s12=eta*Wi; Ds=np.hypot(N1,2*s12); e2=Wi*s12/Ds; c2=Ds/e2
    chi=0.5*np.degrees(np.arctan2(2*s12,N1))
    m=fenep(float(b)); sh=shear_funcs(m,Wi); pl=planar(m,ee)
    print('%3d %5.1f | %.3f %.3f | %.2f %.2f | %.3f %.3f | chi=%.2f | %.3f %.2f  %.3f %.3f | %.2f  %.3f'%(b,Wi,ee,e2,c,c2,r,c/t,chi,sh['s12']/Wi,sh['N1'],sh['N1']/N1,sh['s12']/Wi/eta,pl['P1'],pl['P1']/t))
# alternative FENE-P normalization (Bird: f=1/(1-trC/b), eq. C_eq with (b+5)/b factor) not explored.
# Hookean check
print('Hookean Wi=1: composite',4*(1+1),' eeff',1/(2*np.sqrt(2)),' planar',4/(1-4/8))
