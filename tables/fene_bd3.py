"""FENE dumbbells in steady shear: are the second moment C = <QQ> and the stress tau = <QQ/(1-Q^2/b)> - I coaxial?
Also the Peterlin estimate of the stress from the measured C.  Units lambda_H = nkT = 1."""
import numpy as np, json
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figure_code'))
from fene_bd import corrector_root as corr_root
def run(g, b=50.0, N=30000, dt=0.004, teq=20.0, tav=30.0, seed=3, nblk=15):
    rng = np.random.default_rng(seed); K = np.array([[0, g, 0], [0, 0, 0], [0, 0, 0.]])
    Q = rng.standard_normal((N, 3)); r = np.linalg.norm(Q, axis=1); bad = r*r > 0.9*b; Q[bad] *= (np.sqrt(0.5*b)/r[bad])[:, None]
    nst = int(tav/dt); per = nst//nblk; acc = np.zeros((nblk, 10)); sq = np.sqrt(dt)
    for i in range(-int(teq/dt), per*nblk):
        dW = rng.standard_normal((N, 3))*sq; KQ = Q@K.T; FQ = Q/(1 - np.einsum('ij,ij->i', Q, Q)/b)[:, None]
        Qp = Q + (KQ - 0.5*FQ)*dt + dW; rp = np.einsum('ij,ij->i', Qp, Qp); bad = rp >= b*0.999
        if bad.any(): Qp[bad] *= (np.sqrt(0.999*b/rp[bad]))[:, None]
        rhs = Q + 0.5*(Qp@K.T + KQ - 0.5*FQ)*dt + dW; Rn = np.sqrt(np.einsum('ij,ij->i', rhs, rhs)); Q = rhs*(corr_root(Rn, b, dt)/Rn)[:, None]
        if i >= 0:
            q2 = np.einsum('ij,ij->i', Q, Q); w = 1/(1 - q2/b); x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
            acc[i//per] += [np.mean(w*x*y), np.mean(w*x*x), np.mean(w*y*y), np.mean(w*z*z), np.mean(x*y), np.mean(x*x), np.mean(y*y), np.mean(z*z), np.mean(q2)/b, np.mean(q2 > 0.5*b)]
    return acc/per
def case(g):
    dt = 0.004 if g <= 10 else (0.002 if g <= 30 else 0.001); a = run(g, dt=dt); n = np.sqrt(len(a))
    def derived(m):
        chi_s = 0.5*np.degrees(np.arctan2(2*m[0], m[1]-m[2])); chi_c = 0.5*np.degrees(np.arctan2(2*m[4], m[5]-m[6]))
        f = 1/(1 - (m[5]+m[6]+m[7])/50.0)                      # Peterlin factor from the measured C
        return [chi_s, chi_c, f*m[4], f*(m[5]-m[6]), m[0], m[1]-m[2], m[8], m[9]]
    d = derived(a.mean(0)); se = (np.array([derived(r) for r in a]).std(0, ddof=1)/n).tolist()
    # exact steady moment identity tau = L.C + C.L^T  ->  tau_12 = g*C_22, N1 = 2 g C_12
    chk = [a[:, 0].mean()/(g*a[:, 6].mean()), (a[:, 1]-a[:, 2]).mean()/(2*g*a[:, 4].mean())]
    print(f'Wi={g}: chi_sigma={d[0]:.2f}+-{se[0]:.2f} deg, chi_C={d[1]:.2f}+-{se[1]:.2f} deg | Peterlin(C): S={d[2]:.3f}, N1={d[3]:.3f} vs true S={d[4]:.3f}, N1={d[5]:.3f} | <Q^2>/b={d[6]:.3f}, frac(Q^2>b/2)={d[7]:.4f} | moment identity {chk[0]:.4f}, {chk[1]:.4f}', flush=True)
    return dict(g=g, d=d, se=se, chk=chk)
if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(2) as p: out = p.map(case, [100.0, 30.0, 10.0, 3.0, 1.0, 0.3], chunksize=1)
    json.dump(out, open('fene_bd3.json', 'w'), default=float)
