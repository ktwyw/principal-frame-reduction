import numpy as np, sys
from scipy.integrate import solve_ivp
exec(open('i2_bound.py').read().split('worst={}')[0].replace('def run(','def _unused(') )
rng=np.random.default_rng(1)
bad=0
for trial in range(12):
    K=rng.integers(2,11); dec=rng.uniform(0.5,4)
    tau=10**rng.uniform(-dec,0,K); tau[np.argmax(tau)]=1.0
    G=rng.uniform(0.05,1,K); G=G/(G*tau).sum()
    e0=10**rng.uniform(np.log10(0.02),np.log10(5))
    T=min(40.0,12/e0)
    nu=1/tau
    def f(t,y):
        Sk=y[:K];Nk=y[K:2*K]; S=Sk.sum();N=Nk.sum(); g=e0*np.sqrt(4+(N/S)**2)
        return np.concatenate([G*g-nu*Sk,2*g*Sk-nu*Nk])
    t0=1e-6;g0=2*e0
    y0=np.concatenate([G*g0*t0,G*g0**2*t0**2])
    ts=np.linspace(t0,T,2001)[1:]
    sol=solve_ivp(f,[t0,T],y0,method='LSODA',rtol=1e-12,atol=1e-30,t_eval=ts)
    Sk=sol.y[:K];Nk=sol.y[K:]
    S=Sk.sum(0);N=Nk.sum(0);D=np.hypot(N,2*S)
    Yk=(2*Sk*N-2*S*Nk)/D
    worst=-np.inf
    for i in range(len(ts)):
        for lag in np.r_[0,np.logspace(-3,3,40)]:
            v=(np.exp(-nu*lag)*Yk[:,i]).sum()/np.abs(Yk[:,i]).sum()
            if v>worst: worst=v; wi=(ts[i],lag)
    # single crossing check
    o=np.argsort(nu); sc=all(np.sum(np.diff(np.sign(Yk[o,i]))!=0)<=1 for i in range(len(ts)))
    print(K, f"{e0:.3f}", f"max normalized Ysum={worst:.2e} at {wi}", "single-crossing",sc)
