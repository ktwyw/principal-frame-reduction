import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from rlib import *
from n5_protocol import protocol, planar_startup
m=ptt_lin(0.1); gd=5.0
L=np.zeros((3,3)); L[0,1]=gd
te=np.concatenate([np.linspace(1e-4,0.2,400),np.linspace(0.2,40,8000)[1:]])
s=solve_ivp(lambda t,v: pack(m['rhs'](unpack(v),L)),[0,40],pack(m['A0']),method='Radau',rtol=1e-11,atol=1e-15,t_eval=te)
s12=s.y[1]; N1=s.y[0]-s.y[3]; Ds=np.hypot(N1,2*s12); ee=gd*s12/Ds
einf=ee[-1]; print('terminal eeff',einf,' eeff(0+)',ee[0])
pl=planar_startup([(m,1.,1.)],einf,40,te)
etaP=pl/einf
H_pl=einf*te
H_sh=np.concatenate([[0],cumulative_trapezoid(ee,te)])+ee[0]*te[0]
for lab,comp,Hs in (('Ds/eeff(t), H=int eeff dt',Ds/ee,H_sh),('Ds/eeff_inf, H=int eeff dt',Ds/einf,H_sh),('Ds/eeff(t), H=eeff_inf*t (equal time)',Ds/ee,H_pl),('Ds/eeff_inf, equal time',Ds/einf,H_pl),('Ds/eeff(t), H=gd*t/2',Ds/ee,gd*te/2)):
    # interpolate planar at shear Hencky strain
    ok=(Hs<=H_pl[-1])
    ref=np.interp(Hs[ok],H_pl,etaP)
    r=comp[ok]/ref
    i=np.argmax(np.abs(r-1)); j=np.argmax(np.abs(1/r-1))
    print('%-40s max|comp/planar-1|=%.3f at H=%.3f ; max|planar/comp-1|=%.3f at H=%.3f; range of ratio [%.3f,%.3f]'%(lab,abs(r[i]-1),Hs[ok][i],abs(1/r[j]-1),Hs[ok][j],r.min(),r.max()))
# UCM e0=1 shear strain at Hencky 6, and e0 = 0.6..
for name,mm,e0 in (('UCM',ucm(),1.0),('linPTT',ptt_lin(0.1),0.6),('Giesekus',giesekus(0.2),0.6),('FENE-P',fenep(50.),0.6)):
    t,D,gam,g=protocol([(mm,1.,1.)],e0,6/e0,dense=np.array([6/e0]))
    print(name,e0,'shear strain at Hencky 6:',gam[-1])
# fine check of t^4 law
mw=[(ucm(),0.5/0.2,0.2),(ucm(),0.5/1.8,1.8)]
te2=np.array([0.05,0.1,0.15,0.2])
t,D,gam,g=protocol(mw,0.2,0.2,dense=te2); pl2=planar_startup(mw,0.2,0.2,te2)
print('(1-ratio)*120/(e0^2 t^4) ->',(1-D/pl2)*120/(0.04*te2**4),' expected VarG=1.7778')
