# Independent referee library: single-tensor models, steady states from rest.
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
I3=np.eye(3)
iu=np.triu_indices(3)
def pack(A): return A[iu]
def unpack(v):
    A=np.zeros((3,3)); A[iu]=v; return A+A.T-np.diag(np.diag(A))
def DW(L): return 0.5*(L+L.T),0.5*(L-L.T)
def gs(A,L,xi):
    D,W=DW(L); return W@A-A@W+xi*(D@A+A@D)
# each model: dict(rhs(A,L), stress(A), A0)
def ucm(xi=1.0):
    return dict(rhs=lambda S,L: gs(S,L,xi)-S+2*DW(L)[0], stress=lambda S:S, A0=np.zeros((3,3)))
def ptt_lin(eps=0.1,xi=1.0):
    return dict(rhs=lambda S,L: gs(S,L,xi)-(1+eps*np.trace(S))*S+2*DW(L)[0], stress=lambda S:S, A0=np.zeros((3,3)))
def ptt_exp(eps=0.1):
    return dict(rhs=lambda S,L: gs(S,L,1.0)-np.exp(eps*np.trace(S))*S+2*DW(L)[0], stress=lambda S:S, A0=np.zeros((3,3)))
def giesekus(al=0.2):
    return dict(rhs=lambda S,L: gs(S,L,1.0)-S-al*S@S+2*DW(L)[0], stress=lambda S:S, A0=np.zeros((3,3)))
def fenep(b=50.0):
    f=lambda C:1.0/(1-np.trace(C)/b)
    C0=I3*b/(b+3)
    return dict(rhs=lambda C,L: gs(C,L,1.0)-(f(C)*C-I3), stress=lambda C:f(C)*C-I3, A0=C0)
def roliepoly(tdR=5.0,beta=0.1,delta=-0.5):
    # tau_d=1, tau_R=1/tdR ; conformation A, stress = A - I
    def rhs(A,L):
        tr=np.trace(A)
        return gs(A,L,1.0)-(A-I3)-2*tdR*(1-np.sqrt(3/tr))*(A+beta*(tr/3)**delta*(A-I3))
    return dict(rhs=rhs, stress=lambda A:A-I3, A0=I3.copy())
def steady(model,L,T=400.0,A0=None,polish=True,rtol=1e-11):
    f=lambda t,v: pack(model['rhs'](unpack(v),L))
    v0=pack(model['A0'] if A0 is None else A0)
    sol=solve_ivp(f,[0,T],v0,method='Radau',rtol=rtol,atol=1e-13)
    v=sol.y[:,-1]
    if polish:
        v2,info,ier,msg=fsolve(lambda v:f(0,v),v,xtol=1e-14,full_output=True)
        if ier==1 and np.linalg.norm(v2-v)<1e-5*(1+np.linalg.norm(v)): v=v2
    res=np.linalg.norm(f(0,v))
    return unpack(v),res
def shear_funcs(model,gd,**kw):
    L=np.zeros((3,3)); L[0,1]=gd
    A,res=steady(model,L,**kw); S=model['stress'](A)
    s12=S[0,1];N1=S[0,0]-S[1,1];N2=S[1,1]-S[2,2]
    Ds=np.hypot(N1,2*s12); ee=gd*s12/Ds
    return dict(s12=s12,N1=N1,N2=N2,Ds=Ds,ee=ee,P1=Ds/ee,P2=Ds*(Ds-N1-2*N2)/(2*gd*s12),A=A,S=S,res=res)
def planar(model,ed,**kw):
    L=np.diag([ed,-ed,0.0])
    A,res=steady(model,L,**kw); S=model['stress'](A)
    return dict(P1=(S[0,0]-S[1,1])/ed,P2=(S[2,2]-S[1,1])/ed,A=A,S=S,res=res)
def eigproj(model,A,L):
    """eigen-decomposition of state; d_i = n_i.D.n_i ; principal stresses"""
    w,V=np.linalg.eigh(A)
    D=DW(L)[0]
    d=np.array([V[:,i]@D@V[:,i] for i in range(3)])
    S=model['stress'](A)
    sp_=np.array([V[:,i]@S@V[:,i] for i in range(3)])
    return w,V,d,sp_
