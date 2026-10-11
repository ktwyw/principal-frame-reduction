import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma as Gf
al=0.5
s=np.arange(np.log(1e-10),np.log(1e9),0.08); nu=np.exp(s); G=nu**al*0.08/Gf(al)
tt=np.array([1e-3,1e-2,0.1,1.0]); print('Prony check', [ (G*np.exp(-nu*x)).sum()/x**-al for x in tt])
K=len(G)
def f(t,y):
    Sk=y[:K];Nk=y[K:2*K];xp=y[2*K:3*K];zp=y[3*K:]
    S=Sk.sum();N=Nk.sum(); g=np.sqrt(4+(N/S)**2)
    return np.concatenate([G*g-nu*Sk,2*g*Sk-nu*Nk,2*(zp+2*G)-nu*xp,2*xp-nu*zp])
t0=1e-9; y0=np.concatenate([G*2*t0,G*4*t0**2,4*G*t0,4*G*t0**2])
# better init: exact for tiny t with nu large: use modes' exact constant-rate solution
g0=2.0
Sk0=G*g0*(1-np.exp(-nu*t0))/nu
Nk0=2*g0**2*G*((1-np.exp(-nu*t0))/nu**2 - t0*np.exp(-nu*t0)/nu)  # approx with gd const
y0=np.concatenate([Sk0,Nk0,4*G*(1-np.exp(-nu*t0))/nu,np.zeros(K)])
def jac(t,y): return None
sol=solve_ivp(f,[t0,3.0],y0,method='LSODA',rtol=1e-9,atol=1e-16,t_eval=[0.5,1,2,3],dense_output=False)
for i,x in enumerate(sol.t):
    y=sol.y[:,i]; S=y[:K].sum();N=y[K:2*K].sum(); D=np.hypot(N,2*S)
    Dp_exact=4*np.sum(G*0)  # placeholder
    Dp=y[2*K:3*K].sum()
    # exact planar for power law: 4 int_0^t u^-a cosh 2u du
    from scipy.integrate import quad
    Dpe=4*quad(lambda u: u**-al*np.cosh(2*u),0,x)[0]
    print(x, D/Dp, D/Dpe, 1-8/135*x**2, 'gd/2=',np.sqrt(4+(N/S)**2)/2, 1+2*(1/3)**2*x**2)
