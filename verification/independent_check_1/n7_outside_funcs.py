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
