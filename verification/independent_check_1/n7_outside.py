import numpy as np, sys
from rlib import *
def multi_shear(modes,gd,etas=0.0):
    """modes: list of (model, eta_k, tau_k). returns total S"""
    S=np.zeros((3,3))
    for m,eta,tau in modes:
        L=np.zeros((3,3)); L[0,1]=gd*tau
        A,res=steady(m,L); S+=eta/tau*m['stress'](A)
    S[0,1]+=etas*gd; S[1,0]+=etas*gd
    return S
def comp(S,gd):
    s12=S[0,1];N1=S[0,0]-S[1,1];N2=S[1,1]-S[2,2];Ds=np.hypot(N1,2*s12);ee=gd*s12/Ds
    return ee,Ds/ee,Ds*(Ds-N1-2*N2)/(2*gd*s12)
def multi_planar(modes,ed,etas=0.0):
    S=np.zeros((3,3))
    for m,eta,tau in modes:
        L=np.diag([ed*tau,-ed*tau,0.])
        A,res=steady(m,L); S+=eta/tau*m['stress'](A)
    return (S[0,0]-S[1,1])/ed+4*etas,(S[2,2]-S[1,1])/ed+2*etas
spec=[(0.5,0.2),(0.5,1.8)]; var=0.64
print('=== leading-order tests, two-mode spectrum')
for name,mk in (('linPTT',lambda:ptt_lin(0.1)),('expPTT',lambda:ptt_exp(0.1)),('Giesekus',lambda:giesekus(0.2)),('UCM',lambda:ucm())):
    modes=[(mk(),e,t) for e,t in spec]
    for gd in (0.02,0.05,0.1,0.2):
        S=multi_shear(modes,gd); ee,c1,c2=comp(S,gd); p1,p2=multi_planar(modes,ee)
        print(name,'ee=%.5f (P1-[P1])/(16Var e^2)=%.4f  (P2-[P2])/(8Var e^2)=%.4f   Maxwell-pred for P2: %.4f'%(ee,(p1-c1)/(16*var*ee**2),(p2-c2)/(8*var*ee**2),1-2*ee*(2.92-1)/var))
    sys.stdout.flush()
print('=== solvent subtraction: Giesekus(0.2)+solvent etas=0.5 (etap=1)')
m=giesekus(0.2); etas=0.5
for gd in (0.5,2,10,50):
    S=multi_shear([(m,1.,1.)],gd,etas)
    ee,c1,c2=comp(S,gd)
    p1,p2=multi_planar([(m,1.,1.)],ee,etas)
    # subtraction
    s12p=S[0,1]-etas*gd; N1=S[0,0]-S[1,1]; N2=S[1,1]-S[2,2]; Dp=np.hypot(N1,2*s12p); eep=gd*s12p/Dp
    q1=4*etas+Dp/eep; q2=2*etas+Dp*(Dp-N1-2*N2)/(2*gd*s12p)
    t1,t2=multi_planar([(m,1.,1.)],eep,etas)
    print('gd=%g naive [P1]/P1=%.4f ; subtracted: rel err P1 %.1e P2 %.1e'%(gd,c1/p1,abs(q1/t1-1),abs(q2/t2-1)))
print('=== PTT two-mode high-rate plateau factor (pred 0.8905)')
modes=[(ptt_lin(0.1),e,t) for e,t in spec]
for gd in (1e2,1e3,1e4,1e5,1e6):
    S=multi_shear(modes,gd); ee,c1,c2=comp(S,gd); p1,p2=multi_planar(modes,ee)
    s12k=[]
    print('gd=%g ee=%.4g [P1]=%.4f P1=%.4f ratio=%.4f ; [P1]/(2/eps)=%.4f'%(gd,ee,c1,p1,c1/p1,c1/20))
    sys.stdout.flush()
# single-mode asymptotics
m=ptt_lin(0.1)
for gd in (1e4,1e6):
    sh=shear_funcs(m,gd); print('single PTT gd=%g s12/asym=%.5f N1/asym=%.5f'%(gd,sh['s12']/((2*0.1)**(-1/3)*gd**(1/3)),sh['N1']/((0.01/2)**(-1/3)*gd**(2/3))))
print('=== PTT pair with different eps')
pair=[(ptt_lin(0.080),0.47,2.03),(ptt_lin(0.0015),0.53,0.094)]
best=0
for gd in np.logspace(-1,5,49):
    S=multi_shear(pair,gd); ee,c1,c2=comp(S,gd); p1,p2=multi_planar(pair,ee)
    best=max(best,c1/p1)
    print('gd=%.3g ee=%.4g ratio=%.4f'%(gd,ee,c1/p1))
print('max ratio',best)
