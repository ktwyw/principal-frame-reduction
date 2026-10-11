import numpy as np, sys
from scipy.integrate import solve_ivp, cumulative_trapezoid
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 1)
def run(G,tau,e0,T,npts=4001):
    nu=1/tau; K=len(G)
    def gdot(y):
        S=y[:K].sum(); N=y[K:2*K].sum()
        return e0*np.sqrt(4+(N/S)**2)
    def f(t,y):
        Sk=y[:K];Nk=y[K:2*K]; g=gdot(y)
        xp=y[2*K+1:3*K+1]; zp=y[3*K+1:4*K+1]
        return np.concatenate([G*g-nu*Sk, 2*g*Sk-nu*Nk,[g],
                               2*e0*(zp+2*G)-nu*xp, 2*e0*xp-nu*zp])
    t0=1e-8; g0=2*e0
    # small-time init; planar from exact solution
    y0=np.concatenate([G*g0*t0, G*g0**2*t0**2,[g0*t0], 4*G*e0*t0, 4*G*e0**2*t0**2])
    ts=np.linspace(t0,T,npts)
    sol=solve_ivp(f,[t0,T],y0,method='LSODA',rtol=1e-11,atol=1e-14,t_eval=ts)
    Y=sol.y; Sk=Y[:K];Nk=Y[K:2*K]; gam=Y[2*K]; xp=Y[2*K+1:3*K+1]
    S=Sk.sum(0);N=Nk.sum(0);D=np.sqrt(N**2+4*S**2)
    g=e0*D/S
    Sdot=(G[:,None]*g-nu[:,None]*Sk).sum(0)
    w=(N*(g*N/2+Sdot)+S*(nu[:,None]*Nk).sum(0))/D**2
    # check w against finite difference chi
    chi=0.5*np.arctan2(2*S,N); wfd=g/2+np.gradient(chi,ts)
    h=g*N/(2*D)
    Yk=(2*Sk*N-2*S*Nk)/D
    Dp=xp.sum(0)
    # identity integral at selected times
    res=[]
    for i in range(200,npts,400):
        s=ts[:i+1]; lag=ts[i]-s
        integ=2*(w[:i+1]*np.cosh(2*e0*lag)+h[:i+1]*np.sinh(2*e0*lag))*(np.exp(-np.outer(nu,lag))*Yk[:,:i+1]).sum(0)
        I=np.trapezoid(integ,s)
        res.append((ts[i],D[i]-Dp[i],I,D[i]/Dp[i]))
    # Ysum <=0 for all lags at sampled s
    ysmax=-np.inf
    for i in range(0,npts,200):
        for lag in [0.1,1,10,100]:
            v=(np.exp(-nu*lag/ (1))*Yk[:,i]).sum()/max(1e-300,np.abs(Yk[:,i]).sum()+1e-300)
            ysmax=max(ysmax,v)
    dg=np.diff(g)
    out=dict(mono=dg.min()/g.max(), wmin=w[1:].min(), wfderr=np.max(np.abs(w-wfd)[5:-5])/np.max(np.abs(w)),
             hmin=h.min(), gam_vs_sinh=np.max(gam-2*np.sinh(e0*ts)), ratio_max=np.max(D/Dp), ysum=ysmax,
             ratio_mono=np.diff(D/Dp)[50:].max(), res=res)
    return out
worst={}
for trial in range(int(sys.argv[2]) if len(sys.argv)>2 else 20):
    K=rng.integers(2,11); dec=rng.uniform(0.5,4)
    tau=10**rng.uniform(-dec,0,K); tau[np.argmax(tau)]=1.0
    G=rng.uniform(0.05,1,K); G=G/(G*tau).sum()
    e0=10**rng.uniform(np.log10(0.02),np.log10(5))
    T=min(8.0/e0, 40.0) if e0>0.5 else 40.0
    T=min(T,12/e0)
    o=run(G,tau,e0,T)
    idm=max(abs(r[1]-r[2])/max(abs(r[1]),1e-12) for r in o['res'])
    print(f"K={K} dec={dec:.2f} e0={e0:.3f} mono={o['mono']:.2e} wmin={o['wmin']:.3e} wfd={o['wfderr']:.1e} gam-2sinh={o['gam_vs_sinh']:.2e} ratiomax={o['ratio_max']:.12f} Ysum={o['ysum']:.2e} idrelerr={idm:.1e} diffsign={max(r[1] for r in o['res']):.2e} ratio_inc={o['ratio_mono']:.1e}")
