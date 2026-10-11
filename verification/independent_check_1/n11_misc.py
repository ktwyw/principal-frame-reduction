import numpy as np
from scipy.optimize import brentq
from rlib import *
import n7_outside_funcs as nf
# Giesekus / PTT two-mode P2 scaled diff at exactly ee*taubar=0.01
spec=[(0.5,0.2),(0.5,1.8)]; var=0.64
for name,mk in (('linPTT',lambda:ptt_lin(0.1)),('Giesekus',lambda:giesekus(0.2)),('expPTT',lambda:ptt_exp(0.1))):
    modes=[(mk(),e,t) for e,t in spec]
    g=brentq(lambda g: nf.comp(nf.multi_shear(modes,g),g)[0]-0.01,0.015,0.03,xtol=1e-13)
    S=nf.multi_shear(modes,g); ee,c1,c2=nf.comp(S,g); p1,p2=nf.multi_planar(modes,ee)
    print(name,'ee=%.6f P2 scaled %.4f  P1 scaled %.4f'%(ee,(p2-c2)/(8*var*ee**2),(p1-c1)/(16*var*ee**2)))
# analytic linear PTT multi-mode scan, common eps
def ptt_shear(eta,tau,eps,gd):
    s12=0;N1=0
    for e,t in zip(eta,tau):
        Wi=gd*t; r=np.roots([2*eps,0,1,-Wi]); x=r[np.abs(r.imag)<1e-9].real; x=x[x>0][0]
        G=e/t; s12+=G*x; N1+=G*2*x*x
    Ds=np.hypot(N1,2*s12); ee=gd*s12/Ds; return ee,Ds/ee
def ptt_planar(eta,tau,eps,ed):
    tot=0
    for e,t in zip(eta,tau):
        w=ed*t
        f=brentq(lambda f: f-1-eps*8*w*w/(f*f-4*w*w),2*w+1e-14,2*w+10+100*w)
        tot+=(e/t)*(2*w/(f-2*w)+2*w/(f+2*w))
    return tot/ed
# validate against ODE solver once
m=ptt_lin(0.1); sh=shear_funcs(m,3.0); print('validate',sh['ee'],sh['P1'],ptt_shear([1],[1],0.1,3.0), planar(m,0.7)['P1'],ptt_planar([1],[1],0.1,0.7))
rng=np.random.default_rng(1)
lo=1e9;hi=0
for it in range(600):
    n=rng.integers(2,5); tau=10**rng.uniform(-1.5,1.5,n); p=rng.dirichlet(np.ones(n)); eps=0.1
    tb=np.sum(p*tau); tau=tau/tb
    for gd in np.logspace(-1,5,40):
        ee,c=ptt_shear(p,tau,eps,gd); t=ptt_planar(p,tau,eps,ee); r=c/t
        lo=min(lo,r);hi=max(hi,r)
print('random PTT scan common eps=0.1: ratio range',lo,hi)
