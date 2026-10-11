import numpy as np
from rlib import *
from scipy.optimize import brentq
print('=== orientation formula (conformation tensor) xi=1')
for name,m,conf in (('UCM',ucm(),lambda S:S+I3),('Giesekus',giesekus(0.2),lambda S:S+I3),('linPTT',ptt_lin(0.1),lambda S:S+I3),('FENE-P',fenep(50.),None)):
    for gd in [0.5,2,10]:
        sh=shear_funcs(m,gd)
        C=conf(sh['S']) if conf else sh['A']
        w=np.linalg.eigvalsh(C[:2,:2]); a2,a1=w
        cos2=sh['N1']/sh['Ds']
        print(name,gd,'cos2chi %.10f vs %.10f ; ee/gd %.10f vs %.10f'%(cos2,(a1-a2)/(a1+a2),sh['ee']/gd,np.sqrt(a1*a2)/(a1+a2)), ' 1/c1+1/c2=%.10f'%(1/a1+1/a2), ' I2/I1 (stress, in-plane)=%.6f'%(np.linalg.det(sh['S'][:2,:2])/np.trace(sh['S'])))
# with stress eigenvalues instead (to show the formula fails if a_i = stress eigenvalues)
sh=shear_funcs(ucm(),2.0); w=np.linalg.eigvalsh(sh['S'][:2,:2]); print('UCM using stress eigenvalues: (a1-a2)/(a1+a2)=',(w[1]-w[0])/(w[1]+w[0]),' vs cos2chi',sh['N1']/sh['Ds'])
# Poole relation for mixed flows: JS xi=0.6, PTT
for name,m,xi in (('JS0.6',ucm(0.6),0.6),('linPTT',ptt_lin(0.1),1.0),('expPTT',ptt_exp(0.1),1.0),('Giesekus',giesekus(0.2),1.0)):
    for al in (0.0,0.5,-0.3):
        L=np.zeros((3,3)); L[0,1]=1.0; L[1,0]=al
        A,res=steady(m,L); S=m['stress'](A)
        I1=np.trace(S); I2=0.5*(I1**2-np.trace(S@S))
        print(name,'alpha',al,'I2/I1=%.8f  -1/(2xi)=%.8f  s33=%.2e'%(I2/I1,-1/(2*xi),S[2,2]))
# Giesekus P2 at exactly eps*taubar=0.01 two-mode
