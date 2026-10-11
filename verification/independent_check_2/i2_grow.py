import numpy as np
exec(open('i2_bound.py').read().split('worst={}')[0])
for e0,tau,G in [(0.6,np.array([1,0.1]),np.array([0.5,0.5])),(0.55,np.array([1,0.01,0.001]),np.array([0.01,1,10])),(0.51,np.array([1.,0.05]),np.array([0.02,20.]))]:
    o=run(G,tau,e0,60.0,npts=6001)
    print(e0,[(round(r[0],1),'%.3e'%(r[1]),'%.4f'%r[3]) for r in o['res'][::3]])
    sol=None
