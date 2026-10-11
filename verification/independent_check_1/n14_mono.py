import numpy as np
from scipy.integrate import solve_ivp
rng=np.random.default_rng(3)
def run(G,tau,e0,T):
    n=len(G); t0=1e-6
    def f(t,y):
        s=y[:n]; N=y[n:]
        S=s.sum(); NN=N.sum(); gd=e0*np.hypot(NN,2*S)/S
        return np.concatenate([G*gd-s/tau, 2*gd*s-N/tau])
    g0=2*e0
    y0=np.concatenate([G*g0*t0, G*(g0*t0)**2])
    te=np.logspace(-3,np.log10(T),400); te[-1]=T
    sol=solve_ivp(f,[t0,T],y0,method='Radau',rtol=1e-11,atol=1e-16,t_eval=te)
    Ds=np.hypot(sol.y[n:].sum(0),2*sol.y[:n].sum(0))
    pl=np.zeros_like(te)
    for g,t in zip(G,tau):
        a=1/t-2*e0; b=1/t+2*e0
        pl+=2*e0*g*(1-np.exp(-a*te))/a+2*e0*g*(1-np.exp(-b*te))/b
    return te,Ds/pl
bad=0; worst=0
for it in range(40):
    n=rng.integers(2,5); tau=10**rng.uniform(-1,1,n); eta=rng.dirichlet(np.ones(n)); tb=np.sum(eta*tau); tau/=tb; G=eta/tau
    e0=rng.uniform(0.05,0.95)/(2*tau.max())
    te,r=run(G,tau,e0,60*tau.max())
    steady=(4/(1-4*e0**2))/np.sum(4*eta/(1-4*tau**2*e0**2))
    inc=np.max(np.diff(r))
    worst=max(worst,inc)
    if inc>1e-8 or r.max()>1+1e-8: bad+=1
    if it<5: print('n=%d e0*taumax=%.2f final ratio %.5f steady pred %.5f max increase %.1e'%(n,e0*tau.max(),r[-1],steady,inc))
print('non-monotone or >1 cases:',bad,'of 40; largest upward step',worst)
