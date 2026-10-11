import numpy as np, sys
from rlib import *
from scipy.optimize import brentq
models={'linPTT':ptt_lin(0.1),'Giesekus':giesekus(0.2),'RoliePoly(noFE)':roliepoly(),'FENE-P':fenep(50.0),'JS0.6':ucm(0.6),'UCM':ucm(1.0)}
def sfree(m,d):
    A,res=steady(m,np.diag(d)); S=m['stress'](A); return np.diag(S),res,A
print('=== mixed planar flows L=gd(e1e2+alpha e2e1)')
for name,m in models.items():
    worst=0
    for al in [-0.5,-0.2,0.0,0.3,0.7,1.0]:
        for gd in [0.3,1.0,3.0]:
            if name=='UCM' and (1+al)*gd*0.5>0.45: continue
            if name=='JS0.6' and (1+al)*gd*0.5*0.6>0.45: continue
            L=np.zeros((3,3)); L[0,1]=gd; L[1,0]=al*gd
            A,res=steady(m,L)
            w,V,d,sp_=eigproj(m,A,L)
            # sort by d descending
            o=np.argsort(-d); d=d[o]; sp_=sp_[o]
            # planar ext at d1: axis1 stretch, axis2 compress, axis 3 neutral -> order d: (d1, 0, -d1)
            pl=planar(m,d[0])
            ref=np.array([pl['S'][0,0],pl['S'][2,2],pl['S'][1,1]])
            err=np.max(np.abs(sp_-ref))/np.max(np.abs(ref))
            worst=max(worst,err, abs(d[1])) 
            if res>1e-8 or pl['res']>1e-8: print('   unsteady',name,al,gd,res,pl['res'])
    print(name,'worst mismatch (planar family)',worst)
    sys.stdout.flush()
print('=== 3D flows of Table 3')
LI=np.zeros((3,3)); LI[0,1]=2; LI+=np.diag([-0.15,-0.15,0.3])
LII=np.array([[0.2,1.5,0.7],[0,-0.3,0.9],[0.4,0,0.1]])
for name,m in models.items():
    for fl,L in (('I',LI),('II',LII)):
        A,res=steady(m,L)
        w,V,d,sp_=eigproj(m,A,L)
        s2,res2,A2=sfree(m,d)
        err=np.max(np.abs(sp_-s2))/np.max(np.abs(sp_))
        o=np.argsort(-d)
        print(name,fl,'d=',np.round(d[o],3),'mismatch %.1e'%err,'res',res,res2)
    sys.stdout.flush()
print('=== uniaxial / biaxial tuned flows')
for name,m in models.items():
    if name in('UCM','JS0.6'): continue
    for gd in [0.3,1,3,10]:
        for sgn,lab in ((-1,'uni'),(1,'biax')):
            def Lof(a):
                L=np.zeros((3,3)); L[0,1]=gd; L+=a*np.diag([-0.5,-0.5,1]); return L
            def e_of(a):
                A,res=steady(m,Lof(a),polish=True)
                w,V,d,sp_=eigproj(m,A,Lof(a))
                # in-plane eigenvectors: those with small 3-component
                ip=[i for i in range(3) if abs(V[2,i])<0.5]
                e=max(d[ip])+a/2
                return e,A,d,sp_,V
            g=lambda a: a-sgn*2*e_of(a)[0]/3
            a=brentq(g,-gd,gd,xtol=1e-14,rtol=1e-14)
            e,A,d,sp_,V=e_of(a)
            o=np.argsort(-d); d=d[o]; sp_=sp_[o]
            s2,res2,_=sfree(m,d)
            err=np.max(np.abs((sp_-sp_.mean())-(s2-s2.mean())))/np.max(np.abs(sp_))
            print(name,lab,'gd',gd,'a=%.5f e=%.5f d='%(a,e),np.round(d/e,4),'mismatch %.1e'%err, ' a/gd=%.4f'%(a/gd))
    sys.stdout.flush()
print('=== orientation formula (conformation tensor) xi=1')
for name,m,conf in (('UCM',ucm(),lambda S:S+I3),('Giesekus',giesekus(0.2),lambda S:S+I3),('linPTT',ptt_lin(0.1),lambda S:S+I3),('FENE-P',fenep(50.),None)):
    for gd in [0.5,2,10]:
        sh=shear_funcs(m,gd)
        C=conf(sh['S']) if conf else sh['A']
        w=np.linalg.eigvalsh(C[:2,:2]); a2,a1=w
        cos2=sh['N1']/sh['Ds']
        print(name,gd,'cos2chi %.10f vs %.10f ; ee/gd %.10f vs %.10f'%(cos2,(a1-a2)/(a1+a2),sh['ee']/gd,np.sqrt(a1*a2)/(a1+a2)), ' 1/c1+1/c2=%.10f'%(1/a1+1/a2))
