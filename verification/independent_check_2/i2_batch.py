import numpy as np,sys
src=open('i2_bound.py').read().split('worst={}')[0]
exec(src)
rng=np.random.default_rng(7)
agg=dict(mono=-1,wmin=1e9,gam=-1,ratio=0,inc=-1,idr=0)
for trial in range(int(sys.argv[1])):
    K=rng.integers(2,11); dec=rng.uniform(0.5,4)
    tau=10**rng.uniform(-dec,0,K); tau[np.argmax(tau)]=1.0
    G=rng.uniform(0.05,1,K); G=G/(G*tau).sum()
    e0=10**rng.uniform(np.log10(0.02),np.log10(5))
    T=min(40.0,12/e0)
    o=run(G,tau,e0,T,npts=int(sys.argv[2]))
    idm=max(abs(r[1]-r[2])/max(abs(r[1]),1e-12) for r in o['res'][1:])
    agg['mono']=max(agg['mono'],-o['mono']); agg['wmin']=min(agg['wmin'],o['wmin'])
    agg['gam']=max(agg['gam'],o['gam_vs_sinh']); agg['ratio']=max(agg['ratio'],o['ratio_max'])
    agg['inc']=max(agg['inc'],o['ratio_mono']); agg['idr']=max(agg['idr'],idm)
    if max(r[1] for r in o['res'])>0: print("POSITIVE DIFF",trial)
print(agg)
