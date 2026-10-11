import numpy as np
from rlib import *
from scipy.optimize import brentq
m=fenep(50.)
for gd in [0.3,1,3,10]:
    for sgn,lab in ((-1,'uni'),(1,'biax')):
        def Lof(a):
            L=np.zeros((3,3)); L[0,1]=gd; L+=a*np.diag([-0.5,-0.5,1]); return L
        def e_of(a):
            A,res=steady(m,Lof(a)); w,V,d,sp_=eigproj(m,A,Lof(a))
            ip=[i for i in range(3) if abs(V[2,i])<0.5]
            return max(d[ip])+a/2,d,sp_
        a=brentq(lambda a:a-sgn*2*e_of(a)[0]/3,-gd/2,gd/2,xtol=1e-14,rtol=1e-14)
        e,d,sp_=e_of(a); o=np.argsort(-d); d=d[o]; sp_=sp_[o]
        A2,_=steady(m,np.diag(d)); s2=np.diag(m['stress'](A2))
        print('FENE-P',lab,gd,'d/e',np.round(d/e,4),'mismatch %.1e'%(np.max(abs((sp_-sp_.mean())-(s2-s2.mean())))/np.max(abs(sp_))))
