import numpy as np
from rlib import *
al=0.2; kap=0.2
def mk(rhs): return dict(rhs=rhs,stress=lambda S:S,A0=np.zeros((3,3)))
D_=lambda L: DW(L)[0]
variants={
 'AD2 +': mk(lambda S,L: gs(S,L,1.0)-S-al*S@S+2*D_(L)+kap*(S@D_(L)@D_(L)+D_(L)@D_(L)@S)),
 'AD2 -': mk(lambda S,L: gs(S,L,1.0)-S-al*S@S+2*D_(L)-kap*(S@D_(L)@D_(L)+D_(L)@D_(L)@S)),
 'rate -S only': mk(lambda S,L: gs(S,L,1.0)-(1+kap*np.trace(D_(L)@D_(L)))*S-al*S@S+2*D_(L)),
 'rate S,S2 (const modulus)': mk(lambda S,L: gs(S,L,1.0)-(1+kap*np.trace(D_(L)@D_(L)))*(S+al*S@S)+2*D_(L)),
 'rate all (const viscosity)': mk(lambda S,L: gs(S,L,1.0)-(1+kap*np.trace(D_(L)@D_(L)))*(S+al*S@S-2*D_(L))),
 'rate 1+k*2trD2 -S only': mk(lambda S,L: gs(S,L,1.0)-(1+2*kap*np.trace(D_(L)@D_(L)))*S-al*S@S+2*D_(L)),
}
LI=np.zeros((3,3)); LI[0,1]=2; LI+=np.diag([-0.15,-0.15,0.3])
LII=np.array([[0.2,1.5,0.7],[0,-0.3,0.9],[0.4,0,0.1]])
for name,m in variants.items():
    for fl,L in (('I',LI),('II',LII)):
        A,res=steady(m,L)
        w,V,d,sp_=eigproj(m,A,L)
        A2,res2=steady(m,np.diag(d)); s2=np.diag(A2)
        dev=lambda x:x-x.mean()
        print('%-28s %s flow p.s.'%(name,fl),np.round(sp_,4),' sf',np.round(s2,4),' mism/maxflow %.3f /maxsf %.3f dev %.3f'%(np.max(abs(sp_-s2))/np.max(abs(sp_)),np.max(abs(sp_-s2))/np.max(abs(s2)),np.max(abs(dev(sp_)-dev(s2)))/np.max(abs(dev(sp_)))))
