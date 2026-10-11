import numpy as np
modes=[(0.5,0.2),(0.5,1.8)]
I=np.eye(3)
def ucm(L,eta,tau):
    # S - tau(L S + S L^T) = eta(L+L^T) ; vectorized row-major: (I⊗I - tau(L⊗I + I⊗L)) vec S
    A=np.eye(9)-tau*(np.kron(L,I)+np.kron(I,L))
    return np.linalg.solve(A,(eta*(L+L.T)).ravel()).reshape(3,3)
def stress(L): return sum(ucm(L,e,t) for e,t in modes)
def princ(L):
    S=stress(L); w,V=np.linalg.eigh(S); D=0.5*(L+L.T)
    d=np.array([V[:,i]@D@V[:,i] for i in range(3)]); return w,d
rng=np.random.default_rng(3)
M=rng.normal(size=(3,3)); np.fill_diagonal(M,0)
for base in [np.array([0.2,-0.15,-0.05]), np.array([0.2,-0.1,-0.1])]:
    print('base',base)
    for de in [0.04,0.02,0.01]:
        L=np.diag(base)+de*M; w,d=princ(L); w0,_=princ(np.diag(d))
        # match by sorting
        err=np.max(np.abs(np.sort(w)-np.sort(w0)))/np.max(np.abs(w0))
        print(f"  delta={de} mismatch={err:.3e} ratio/delta^2={err/de**2:.3f}  d-change={np.max(np.abs(np.sort(d)-np.sort(base))):.2e}")
