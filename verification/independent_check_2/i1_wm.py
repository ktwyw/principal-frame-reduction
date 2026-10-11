import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'independent_check_1')); import numpy as np
from rlib import steady, gs, DW
from scipy.optimize import brentq
def wm(L):
    D=DW(L)[0]; tau=1/(1+np.sqrt(2*np.trace(D@D)))
    return dict(rhs=lambda S,L_: (gs(S,L_,1.0)-S/tau+2*D/tau*1.0), stress=lambda S:S, A0=np.zeros((3,3)))
def flow(gd,al):
    L=np.zeros((3,3));L[0,1]=gd;L[1,0]=al*gd
    m=wm(L); A,res=steady(m,L,T=200)
    w,V=np.linalg.eigh(A[:2,:2]); n=np.r_[V[:,1],0]; d1=n@DW(L)[0]@n
    return w[1]-w[0],d1
def planar(e):
    L=np.diag([e,-e,0.]); m=wm(L); A,res=steady(m,L,T=200); return A[0,0]-A[1,1]
for de in [0.2,0.1,0.05]:
    al=1-de; gd=brentq(lambda g: flow(g,al)[1]-0.2,0.15,0.3,xtol=1e-13)
    Ds,d1=flow(gd,al); m=Ds/planar(d1)-1; print(de,m,m/de**2)
