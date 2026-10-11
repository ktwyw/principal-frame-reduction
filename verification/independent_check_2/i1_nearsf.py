import numpy as np
from scipy.optimize import brentq
modes=[(0.5,0.2),(0.5,1.8)]
def ucm_steady(L,eta,tau):
    # S - tau(L S + S L^T) = eta (L+L^T), 2D, solve linear system
    n=2; A=np.zeros((4,4)); b=(eta*(L+L.T)).ravel()
    for i in range(2):
        for j in range(2):
            r=i*2+j; A[r,r]+=1
            for k in range(2):
                A[r,k*2+j]-=tau*L[i,k]   # (L S)_ij
                A[r,i*2+k]-=tau*L[j,k]   # (S L^T)_ij
    return np.linalg.solve(A,b).reshape(2,2)
def flow(gd,al):
    L=np.array([[0,gd],[al*gd,0.]])
    S=sum(ucm_steady(L,e,t) for e,t in modes)
    w,V=np.linalg.eigh(S); n=V[:,1]
    D=0.5*(L+L.T); d1=n@D@n
    return w[1]-w[0],d1
def planar(ed):
    return sum(4*e*ed/(1-4*t*t*ed*ed) for e,t in modes)  # sigma11-sigma22
for de in [0.4,0.2,0.1,0.05,0.025,0.0125]:
    al=1-de
    gmax=0.999/(3.6*np.sqrt(al)) if al>0 else 10; gd=brentq(lambda g: flow(g,al)[1]-0.2,0.01,gmax)
    Ds,d1=flow(gd,al)
    m=Ds/planar(d1)-1
    print(de,gd,m,m/de**2)
