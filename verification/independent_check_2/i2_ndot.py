import numpy as np
from scipy.integrate import quad
nu=0.7; G=1.0; t=2.3
gd=lambda s: 1+s**2+0.3*np.sin(s)**2
gdd=lambda s: 2*s+0.6*np.sin(s)*np.cos(s)
gam=lambda s: quad(gd,0,s)[0]
gu=lambda u: gam(t)-gam(t-u)
S=G*quad(lambda u: np.exp(-nu*u)*gd(t-u),0,t)[0]
N=2*G*quad(lambda u: np.exp(-nu*u)*gd(t-u)*gu(u),0,t)[0]
Ndot=2*gd(t)*S-nu*N
phi=lambda v: 2*gu(v)+2*quad(lambda u: np.exp(-nu*(u-v))*gd(t-u),v,t)[0]
rep=G*(quad(lambda v: np.exp(-nu*v)*phi(v)*gdd(t-v),0,t,limit=200)[0]+np.exp(-nu*t)*phi(t)*gd(0))
Sdot=G*gd(t)-nu*S
repS=G*(quad(lambda v: np.exp(-nu*v)*gdd(t-v),0,t)[0]+np.exp(-nu*t)*gd(0))
print(Ndot,rep,Sdot,repS)
