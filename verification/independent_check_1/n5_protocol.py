import numpy as np, sys
from scipy.integrate import solve_ivp
from rlib import *
def protocol(models_w, e0, T, t0=1e-7, dense=None):
    """models_w: list of (model, weight G, time-scale tau). Each mode: tau dA/dt = rhs(A, tau L). total stress = sum G*stress.
    state = [gamma, modes...]"""
    n=len(models_w)
    def total(v):
        S=np.zeros((3,3))
        for k,(m,G,tau) in enumerate(models_w):
            S+=G*m['stress'](unpack(v[1+6*k:7+6*k]))
        return S
    def gdot(v):
        S=total(v); s12=S[0,1]; N1=S[0,0]-S[1,1]
        return e0*np.hypot(N1,2*s12)/s12
    def f(t,v):
        gd=gdot(v); L=np.zeros((3,3)); L[0,1]=gd
        out=[gd]
        for k,(m,G,tau) in enumerate(models_w):
            out.append(pack(m['rhs'](unpack(v[1+6*k:7+6*k]),tau*L))/tau)
        return np.concatenate([np.atleast_1d(o) for o in out])
    # initial: small elastic strain gamma0=2 sinh(e0 t0)
    g0=2*np.sinh(e0*t0); v0=[g0]
    for (m,G,tau) in models_w:
        # short-time: integrate each mode exactly elastic: A = A0 + (response to strain g0); do tiny explicit ivp with constant rate
        L=np.zeros((3,3)); L[0,1]=g0/t0
        s=solve_ivp(lambda t,v: pack(m['rhs'](unpack(v),tau*L))/tau,[0,t0],pack(m['A0']),rtol=1e-12,atol=1e-22)
        v0.append(s.y[:,-1])
    v0=np.concatenate([np.atleast_1d(o) for o in v0])
    sol=solve_ivp(f,[t0,T],v0,method='Radau',rtol=1e-10,atol=1e-14,t_eval=dense)
    Ds=[];gam=sol.y[0];gds=[]
    for i in range(sol.y.shape[1]):
        S=total(sol.y[:,i]); Ds.append(np.hypot(S[0,0]-S[1,1],2*S[0,1])); gds.append(gdot(sol.y[:,i]))
    return sol.t,np.array(Ds),gam,np.array(gds)
def planar_startup(models_w,e0,T,dense):
    out=np.zeros(len(dense))
    for (m,G,tau) in models_w:
        L=np.diag([e0,-e0,0.])
        s=solve_ivp(lambda t,v: pack(m['rhs'](unpack(v),tau*L))/tau,[0,T],pack(m['A0']),method='Radau',rtol=1e-11,atol=1e-15,t_eval=dense)
        for i in range(len(dense)):
            S=m['stress'](unpack(s.y[:,i])); out[i]+=G*(S[0,0]-S[1,1])
    return out
print('=== protocol vs planar start-up, single-tensor models')
for name,m in (('UCM',ucm()),('linPTT',ptt_lin(0.1)),('Giesekus',giesekus(0.2)),('FENE-P',fenep(50.)),('RoliePoly',roliepoly())):
    for e0 in (1.0,2.0):
        Tend=6/e0 if name!='UCM' else 3/e0
        te=np.linspace(0.05,Tend,40)
        t,Ds,gam,gds=protocol([(m,1.0,1.0)],e0,Tend,dense=te)
        pl=planar_startup([(m,1.0,1.0)],e0,Tend,te)
        print(name,'e0=%g max rel dev %.2e ; shear strain at t=%.2f (Hencky %.1f): %.2f ; gd end %.3f'%(e0,np.max(np.abs(Ds/pl-1)),Tend,e0*Tend,gam[-1],gds[-1]))
        sys.stdout.flush()
print('2 sinh 6 =',2*np.sinh(6))
print('=== two-mode Maxwell, e0*taubar=0.2')
mw=[(ucm(),0.5/0.2,0.2),(ucm(),0.5/1.8,1.8)]
te=np.array([0.02,0.05,0.1,0.2,0.5,1.0,3.0,10.0,30.0,100.0])
t,Ds,gam,gds=protocol(mw,0.2,100.0,dense=te)
pl=planar_startup(mw,0.2,100.0,te)
G=np.array([2.5,0.5/1.8]); r=np.array([5,1/1.8])
VarG=np.sum(G*r*r)/G.sum()-(np.sum(G*r)/G.sum())**2
for i in range(len(te)):
    print('t=%6.2f ratio=%.6f  series 1-e0^2 t^4 Var/120=%.6f  gd=%.5f  series gd=%.5f'%(te[i],Ds[i]/pl[i],1-0.04*te[i]**4*VarG/120,gds[i],2*.2+.2**3*te[i]**2+.2**3/3*(-np.sum(G*r)/G.sum())*te[i]**3))
print('VarG(1/tau)=',VarG)
