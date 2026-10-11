import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import brentq
def ptt_shear(eta,tau,eps,gd):
    s12=0;N1=0
    for e,t in zip(eta,tau):
        Wi=gd*t
        # x(1+2 eps x^2)=Wi  (Cardano-safe: brentq)
        x=brentq(lambda x:x*(1+2*eps*x*x)-Wi,0,Wi)
        G=e/t; s12+=G*x; N1+=G*2*x*x
    Ds=np.hypot(N1,2*s12); ee=gd*s12/Ds; return ee,Ds/ee
def ptt_planar(eta,tau,eps,ed):
    tot=0
    for e,t in zip(eta,tau):
        w=ed*t
        g=lambda u: (2*w+u)-1-eps*8*w*w/(u*(4*w+u))   # f=2w+u
        u=brentq(g,1e-300,10+100*w)
        f=2*w+u
        tot+=(e/t)*(2*w/u+2*w/(f+2*w))
    return tot/ed
rng=np.random.default_rng(7)
for eps in (0.0003,0.001,0.003):
  for span in (0.5,1.5):
    lo=1e9;hi=0;arg=None
    for it in range(600):
        n=rng.integers(2,5); tau=10**rng.uniform(-span,span,n); p=10**rng.uniform(-3,0,n); p/=p.sum()
        tau=tau/np.sum(p*tau)
        for gd in np.logspace(-2,8,60):
            ee,c=ptt_shear(p,tau,eps,gd); t=ptt_planar(p,tau,eps,ee); r=c/t
            if r>hi: hi=r;arg=(p,tau,gd,ee)
            lo=min(lo,r)
    print('eps=%g span=+-%g decades: ratio range %.4f .. %.6f'%(eps,span,lo,hi), ' argmax',arg if hi>1.0001 else '')
