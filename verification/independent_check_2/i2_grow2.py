import numpy as np
from scipy.integrate import solve_ivp
def run(G,tau,e0,T):
    nu=1/tau;K=len(G)
    def f(t,y):
        Sk=y[:K];Nk=y[K:];S=Sk.sum();N=Nk.sum();g=e0*np.sqrt(4+(N/S)**2)
        return np.concatenate([G*g-nu*Sk,2*g*Sk-nu*Nk])
    t0=1e-8;y0=np.concatenate([G*2*e0*t0,G*4*e0**2*t0**2])
    ts=np.linspace(10,T,8)
    sol=solve_ivp(f,[t0,T],y0,method='LSODA',rtol=1e-10,atol=1e-20,t_eval=ts)
    for i,t in enumerate(ts):
        Sk=sol.y[:K,i];Nk=sol.y[K:,i];S=Sk.sum();N=Nk.sum()
        gd=e0*np.sqrt(4+(N/S)**2)
        # planar exact
        Dp=sum(4*G[k]*e0/(1/tau[k]**2-4*e0**2)*(1/tau[k]*0+0) for k in range(K))
        print(f"t={t:.0f} Dshear={np.hypot(N,2*S):.4e} gd={gd:.3e} N1_slow/N={Nk[0]/N:.3f} S_slow/S={Sk[0]/S:.3e}")
run(np.array([0.01,1,10]),np.array([1,0.01,0.001]),0.55,400)
print('two-mode (0.5,1),(0.5,0.1), taubar=0.918, 1/2taubar=0.545; e0=0.52')
run(np.array([0.5,0.5]),np.array([1,0.1]),0.52,400)
print('paper two-mode eta=(0.5,0.5) tau=(0.2,1.8) taubar=1 ; e0=0.4 (>1/3.6=0.278, <0.5)')
run(np.array([0.5/0.2,0.5/1.8]),np.array([0.2,1.8]),0.4,400)
