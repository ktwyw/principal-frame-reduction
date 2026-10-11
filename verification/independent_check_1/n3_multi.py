import numpy as np, mpmath as mp
from scipy.integrate import quad
from scipy.special import zeta
# multimode Maxwell numbers
def P1true(et,taus,ed): return sum(4*e/(1-4*t*t*ed*ed) for e,t in zip(et,taus))
def P1comp(et,taus,ed):
    e0=sum(et); tb=sum(e*t for e,t in zip(et,taus))/e0; return 4*e0/(1-4*tb*tb*ed*ed)
print('single 0.25:',P1true([1],[1],0.25),' two-mode:',P1true([.5,.5],[.2,1.8],0.25))
print('steady ratio at 0.2:',P1comp([.5,.5],[.2,1.8],0.2)/P1true([.5,.5],[.2,1.8],0.2))
# Rouse/Zimm
print('Rouse <tau>/tau1',zeta(4)/zeta(2),np.pi**2/15,' ratio',zeta(4)**2/(zeta(2)*zeta(6)),' div',1/(2*zeta(4)/zeta(2)), ' tau1/taubar',zeta(2)/zeta(4))
print('Zimm ratio',zeta(3)**2/(zeta(1.5)*zeta(4.5)),' div',1/(2*zeta(3)/zeta(1.5)))
p=np.arange(1,200001)
w=(1/p**2)/np.sum(1/p**2)
print('Rouse at 0.4: true',np.sum(4*w/(1-4*0.16/p**4)),' comp',4/(1-4*(np.pi**2/15)**2*0.16))
# discrete Rouse chain with N beads: tau_p = 1/(4 sin^2(p pi/2N)), p=1..N-1
for N in (10,20,50,100):
    pp=np.arange(1,N); tp=1/(4*np.sin(pp*np.pi/(2*N))**2)
    tb=np.sum(tp**2)/np.sum(tp)
    print('N=%d discrete Rouse tau1/taubar=%.4f ; <tau>^2/<tau^2>=%.4f'%(N,tp[0]/tb, tb**2/(np.sum(tp**3)/np.sum(tp))))
for N,tb in ((50,678.),(20,99.)):
    print('N',N,'1/(2*1.52*taubar)=',1/(2*1.52*tb), ' with 15/pi^2:',1/(2*15/np.pi**2*tb))
    pp=np.arange(1,N); tp=1/(4*np.sin(pp*np.pi/(2*N))**2); r=tp[0]/(np.sum(tp**2)/np.sum(tp))
    print('     with discrete-chain ratio %.3f: %.3e'%(r,1/(2*r*tb)))
# P2 scaled difference for Maxwell 2-mode at 0.01
et=[.5,.5];ta=[.2,1.8];ed=0.01
P2t=sum(2*e/(1+2*t*ed) for e,t in zip(et,ta)); P2c=2/(1+2*ed)
var=1.64-1
print('P2 scaled diff at 0.01 (Maxwell):',(P2t-P2c)/(8*var*ed**2), ' P1 scaled:',(P1true(et,ta,ed)-P1comp(et,ta,ed))/(16*var*ed**2))
# White-Metzner
def wm(g):
    tau=1/(1+g); eta=tau; x=tau*g; P1=4*eta*(1+x*x); ee=g/(2*np.sqrt(1+x*x))
    t2=1/(1+2*ee); true=4*t2/(1-4*t2*t2*ee*ee)
    return P1,ee,true
print('WM at 5:',wm(5.0))
gg=np.linspace(0.01,3,3000); r=np.array([wm(g) for g in gg])
print('WM max rel diff for gd<=3: vs planar %.3f, vs composite %.3f'%(np.max(1-r[:,0]/r[:,2]),np.max(r[:,2]/r[:,0]-1)))
# Oldroyd-B check
etas=0.3;etap=0.7;g=2.0
s12=g;N1=2*etap*g*g;Ds=np.hypot(N1,2*s12);ee=g*s12/Ds
print('Oldroyd-B composite',Ds/ee,' formula',4/(1-4*etap**2*ee**2),' true',4*etas+4*etap/(1-4*ee**2) if ee<.5 else None)
# PTT C-S factor
x=np.array([.2,1.8])**(-1/3.); print('CS factor',np.mean(x)**2/np.mean(x*x))
# KBKZ predictions
def mom(pk,tk,n): return sum(p*t**n for p,t in zip(pk,tk))
for pk,tk,a in (([.5,.5],[.2,1.8],.05),([.3,.7],[.5,1.3],.2)):
    c4=64*(mom(pk,tk,4)-mom(pk,tk,1)**4)-384*a*(5*mom(pk,tk,4)-4*mom(pk,tk,1)*mom(pk,tk,3))
    print('KBKZ predicted eps^4 coeff',c4)
# KBKZ direct quadrature (mpmath high precision)
mp.mp.dps=40
def kbkz(pk,tk,a,damp):
    def h(I):
        return 1/(1+a*I) if damp=='psm' else mp.exp(-a*I)
    def m(s): return sum(p/t**2*mp.exp(-s/t) for p,t in zip(pk,tk))
    def shear(g):
        s12=mp.quad(lambda s:m(s)*h(g*g*s*s)*g*s,[0,1,5,20,100,400])
        n1=mp.quad(lambda s:m(s)*h(g*g*s*s)*g*g*s*s,[0,1,5,20,100,400])
        Ds=mp.sqrt(n1**2+4*s12**2); ee=g*s12/Ds
        return ee,Ds/ee
    def plan(e):
        return mp.quad(lambda s:m(s)*h(4*mp.sinh(e*s)**2)*2*mp.sinh(2*e*s),[0,1,5,20,100,400])/e
    return shear,plan
for pk,tk,a in (([.5,.5],[.2,1.8],.05),([.3,.7],[.5,1.3],.2)):
    for damp in ('psm','exp'):
        shear,plan=kbkz(pk,tk,a,damp)
        var=mom(pk,tk,2)-mom(pk,tk,1)**2
        out=[]
        for g in (0.02,0.04,0.08):
            ee,c=shear(mp.mpf(g)); t=plan(ee)
            out.append((float(ee),float((t-c-16*var*ee**2)/ee**4)))
        print(damp,pk,tk,a,'  (eps, [diff-16Var e^2]/e^4):',out)
# single-mode PSM a=0.05: composite above planar by up to 76%?
mp.mp.dps=20
shear,plan=kbkz([1.0],[1.0],0.05,'psm')
best=(0,0)
for g in np.logspace(-0.5,2.5,61):
    ee,c=shear(mp.mpf(g)); t=plan(ee); r=float(c/t)
    if r>best[0]: best=(r,float(ee),g)
    print('g=%.3g ee=%.4f comp=%.4f planar=%.4f ratio=%.4f'%(g,ee,c,t,r))
print('max ratio',best)
