import numpy as np, sys
from rlib import *
models={'UCM':ucm(1.0),'LCM':ucm(-1.0),'JS0.6':ucm(0.6),'linPTT':ptt_lin(0.1),'expPTT':ptt_exp(0.1),'Giesekus':giesekus(0.2),'FENE-P':fenep(50.0),'RoliePoly(noFE)':roliepoly()}
rates=[0.03,0.1,0.3,1,3,10,30,100]
for name,m in models.items():
    worst=0; info=[]
    for gd in rates:
        try:
            sh=shear_funcs(m,gd)
        except Exception as ex:
            info.append((gd,'shear fail')); continue
        if sh['res']>1e-8: info.append((gd,'shear not steady %g'%sh['res'])); continue
        try:
            pl=planar(m,sh['ee'],T=400)
        except Exception as ex:
            info.append((gd,'planar fail ee=%.4g'%sh['ee'])); continue
        if pl['res']>1e-8 or not np.all(np.isfinite(pl['S'])):
            info.append((gd,'planar not steady ee=%.4g res=%g'%(sh['ee'],pl['res']))); continue
        e1=abs(sh['P1']/pl['P1']-1); e2=abs(sh['P2']/pl['P2']-1)
        worst=max(worst,e1,e2)
        info.append((gd,'ee=%.5g P1=%.6g P2=%.6g e1=%.1e e2=%.1e'%(sh['ee'],sh['P1'],sh['P2'],e1,e2)))
    print(name,'max rel diff %.2e'%worst)
    for i in info: print('   ',i)
    sys.stdout.flush()
