import numpy as np
from scipy.integrate import solve_ivp
P=200; p=np.arange(1,P+1); tau=1.0/p**2; nu=1/tau; G=np.ones(P); K=P
eta=G*tau; eta0=eta.sum(); tb=(eta*tau).sum()/eta0
for e in [0.2,0.4]:
    comp=4*eta0/(1-4*tb**2*e**2); true=(4*eta/(1-4*tau**2*e**2)).sum()
    print('steady',e,comp/true, comp/eta0, true/eta0)
def run(e0,xs):
    def f(t,y):
        Sk=y[:K];Nk=y[K:2*K];xp=y[2*K:3*K];zp=y[3*K:]
        S=Sk.sum();N=Nk.sum(); g=e0*np.sqrt(4+(N/S)**2)
        return np.concatenate([G*g-nu*Sk,2*g*Sk-nu*Nk,2*e0*(zp+2*G)-nu*xp,2*e0*xp-nu*zp])
    t0=1e-10;g0=2*e0
    y0=np.concatenate([G*g0*t0,G*g0**2*t0**2,4*G*e0*t0,4*G*e0**2*t0**2])
    ts=np.array(xs)/e0
    sol=solve_ivp(f,[t0,ts[-1]],y0,method='LSODA',rtol=1e-10,atol=1e-20,t_eval=ts)
    out=[]
    for i in range(len(ts)):
        y=sol.y[:,i];S=y[:K].sum();N=y[K:2*K].sum();out.append(np.hypot(N,2*S)/y[2*K:3*K].sum())
    return out
for e0 in [0.2,0.4,1,3,10,30]:
    xs=[0.5,1,2] if e0>=1 else [0.5,1,2,5,10,20,40]
    print(e0, np.round(run(e0,xs),4))
