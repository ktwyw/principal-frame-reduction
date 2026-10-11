import numpy as np, sys
from rlib import *
al=0.2; kap=0.2
G=giesekus(al)
def comm(A,X): return A@X-X@A
def DA_of(A,D):
    w,V=np.linalg.eigh(A)
    return sum((V[:,i]@D@V[:,i])*np.outer(V[:,i],V[:,i]) for i in range(3))
def base(S,L): return gs(S,L,1.0)-S-al*S@S+2*DW(L)[0]
def mk(rhs): return dict(rhs=rhs,stress=lambda S:S,A0=np.zeros((3,3)))
def spin(S,L):
    D,W=DW(L); Wm=W+kap*(S@D-D@S)
    return Wm@S-S@Wm+(D@S+S@D)-S-al*S@S+2*D
terms={
 'modified spin': mk(spin),
 '[A,[A,D^2]]': mk(lambda S,L: base(S,L)+kap*comm(S,comm(S,DW(L)[0]@DW(L)[0]))),
 'trD^2 [A,[A,D]]': mk(lambda S,L: base(S,L)+kap*np.trace(DW(L)[0]@DW(L)[0])*comm(S,comm(S,DW(L)[0]))),
 'D_A^2': mk(lambda S,L: base(S,L)+kap*DA_of(S,DW(L)[0])@DA_of(S,DW(L)[0])),
 'relax 1+k trD_A^2 (on -S only)': mk(lambda S,L: gs(S,L,1.0)-(1+kap*np.trace(DA_of(S,DW(L)[0])@DA_of(S,DW(L)[0])))*S-al*S@S+2*DW(L)[0]),
 'relax 1+k trD_A^2 (on S and S^2)': mk(lambda S,L: gs(S,L,1.0)-(1+kap*np.trace(DA_of(S,DW(L)[0])@DA_of(S,DW(L)[0])))*(S+al*S@S)+2*DW(L)[0]),
 'A.D^2+D^2.A (+)': mk(lambda S,L: base(S,L)+kap*(S@DW(L)[0]@DW(L)[0]+DW(L)[0]@DW(L)[0]@S)),
 'A.D^2+D^2.A (-)': mk(lambda S,L: base(S,L)-kap*(S@DW(L)[0]@DW(L)[0]+DW(L)[0]@DW(L)[0]@S)),
 'relax 1+k trD^2 (on -S only)': mk(lambda S,L: gs(S,L,1.0)-(1+kap*np.trace(DW(L)[0]@DW(L)[0]))*S-al*S@S+2*DW(L)[0]),
 'relax 1+k trD^2 (on S and S^2)': mk(lambda S,L: gs(S,L,1.0)-(1+kap*np.trace(DW(L)[0]@DW(L)[0]))*(S+al*S@S)+2*DW(L)[0]),
}
LI=np.zeros((3,3)); LI[0,1]=2; LI+=np.diag([-0.15,-0.15,0.3])
LII=np.array([[0.2,1.5,0.7],[0,-0.3,0.9],[0.4,0,0.1]])
for name,m in terms.items():
    out=[]
    for fl,L in (('I',LI),('II',LII)):
        # perturb the initial state slightly off isotropy to avoid degenerate eigenvectors at t=0 for D_A terms
        A,res=steady(m,L,A0=1e-6*np.diag([1.,2.,3.])+1e-7*(L+L.T))
        w,V,d,sp_=eigproj(m,A,L)
        A2,res2=steady(m,np.diag(d),A0=1e-6*np.diag([3.,1.,2.]))
        s2=np.diag(A2)
        out.append((fl,np.max(np.abs(sp_-s2))/np.max(np.abs(sp_)),res,res2))
    print('%-36s'%name,' '.join('%s: %.2e (res %.0e,%.0e)'%o for o in out),' max=%.2e'%max(o[1] for o in out))
    sys.stdout.flush()
# realizability numbers
pl=planar(G,0.8); a=np.diag(pl['S'])
print('Giesekus planar ext d1=0.8 principal stresses',a)
for h in (0.3,1.0,3.0):
    W12=h*(a[0]+a[1]+2)/(a[0]-a[1])
    L=np.array([[0.8,h+W12,0],[h-W12,-0.8,0],[0,0,0]])
    A,res=steady(G,L,T=2000)
    print('h=%.1f W12=%.4f ; steady from rest:'%(h,W12),np.round(np.linalg.eigvalsh(A),10),' offdiag',A[0,1],' res',res,' ref',np.sort(a))
# opposite sign convention check
for h in (1.0,):
    W12=-h*(a[0]+a[1]+2)/(a[0]-a[1])
    L=np.array([[0.8,h+W12,0],[h-W12,-0.8,0],[0,0,0]])
    A,res=steady(G,L,T=2000)
    print('wrong-sign W: eig',np.linalg.eigvalsh(A),A[0,1])
