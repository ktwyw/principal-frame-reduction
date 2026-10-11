import numpy as np
from rlib import *
D_=lambda L: DW(L)[0]
def mk(rhs): return dict(rhs=rhs,stress=lambda S:S,A0=np.zeros((3,3)))
def scan(m,rates=np.logspace(np.log10(0.03),np.log10(3),13)):
    w1=w2=0
    for gd in rates:
        sh=shear_funcs(m,gd)
        if sh['res']>1e-8 or sh['ee']<=0: continue
        pl=planar(m,sh['ee'])
        if pl['res']>1e-8 or not np.isfinite(pl['P1']): continue
        w1=max(w1,abs(sh['P1']/pl['P1']-1)); w2=max(w2,abs(sh['P2']/pl['P2']-1))
    return w1,w2
for c in (0.3,0.6,1.2,-0.3,-1.2):
    print('UCM + c D^2, c=%g: max rel diff P1 %.3f P2 %.3f'%((c,)+scan(mk(lambda S,L,c=c: gs(S,L,1.)-S+2*D_(L)+c*D_(L)@D_(L)))))
for c in (0.3,-0.3,1.2,-1.2):
    print('UCM + c (S D^2 + D^2 S), c=%g: P1 %.3f P2 %.3f'%((c,)+scan(mk(lambda S,L,c=c: gs(S,L,1.)-S+2*D_(L)+c*(S@D_(L)@D_(L)+D_(L)@D_(L)@S)))))
for c in (0.02,0.1,0.3):
    print('UCM, relax rate 1+c tr(S D^2), c=%g: P1 %.3f P2 %.3f'%((c,)+scan(mk(lambda S,L,c=c: gs(S,L,1.)-(1+c*np.trace(S@D_(L)@D_(L)))*S+2*D_(L)))))
# inside-class terms
k=0.2
ins={
 'phi1(J1): rate 1+k*S:D': lambda S,L: gs(S,L,1.)-(1+k*np.trace(S@D_(L)))*S+2*D_(L),
 '(trS)D,(S:D)I, xi=0.8': lambda S,L: gs(S,L,.8)-S+2*D_(L)+k*np.trace(S)*D_(L)+k*np.trace(S@D_(L))*I3,
 'SD+DS, S2D+DS2': lambda S,L: gs(S,L,1.)-S+2*D_(L)-k*(S@D_(L)+D_(L)@S)-0.05*(S@S@D_(L)+D_(L)@S@S),
 'strand ext -(S:D)(S+I)': lambda S,L: gs(S,L,1.)-S+2*D_(L)-k*np.trace(S@D_(L))*(S+I3),
}
for n,f in ins.items(): print(n,'P1 %.1e P2 %.1e'%scan(mk(f)))
