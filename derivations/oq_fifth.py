"""Fifth-order fluid: which constants enter eta_P1 - eta_[P1] at O(eps^4)?  Spanning set of isotropic polynomial terms in gamma_(1..5)."""
import sympy as sp, itertools, json
g, e = sp.symbols('gammadot epsilon', positive=True)
E11 = sp.diag(1, 0, 0); P = sp.diag(1, -1, 0); Q = sp.diag(1, 1, 0)
SH = {1: g*sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), 2: -2*g**2*E11, 3: sp.zeros(3), 4: sp.zeros(3), 5: sp.zeros(3)}
PL = {n: 2*e*(-2*e)**(n - 1)*(P if n % 2 else Q) for n in range(1, 6)}
def partitions(n, maxpart=None):
    maxpart = maxpart or n
    if n == 0: yield []; return
    for k in range(min(n, maxpart), 0, -1):
        for p in partitions(n - k, k): yield [k] + p
def set_partitions(items):
    if not items: yield []; return
    first, rest = items[0], items[1:]
    for sp_ in set_partitions(rest):
        yield [[first]] + sp_
        for i in range(len(sp_)): yield sp_[:i] + [[first] + sp_[i]] + sp_[i+1:]
def terms(order):
    seen = {}
    for ms in partitions(order):
        idx = list(range(len(ms)))
        for r in range(1, len(ms) + 1):
            for mat in itertools.combinations(idx, r):
                rest = [i for i in idx if i not in mat]
                for tp in set_partitions(rest):
                    for mperm in set(itertools.permutations([ms[i] for i in mat])):
                        trs = [sorted(set(itertools.permutations([ms[i] for i in grp]))) for grp in tp]
                        for trc in itertools.product(*trs):
                            key = (mperm, tuple(sorted(trc)))
                            seen[key] = 1
    return list(seen)
def val(term, T):
    mperm, trc = term; M = sp.eye(3)
    for n in mperm: M = M*T[n]
    M = (M + M.T)/2; c = 1
    for grp in trc:
        X = sp.eye(3)
        for n in grp: X = X*T[n]
        c = c*X.trace()
    return sp.simplify(c)*M
res = {}
for order in (4, 5):
    for tm in terms(order):
        s = val(tm, SH); p = val(tm, PL)
        res[tm] = dict(order=order, s12=sp.expand(s[0, 1]/g**order), n1=sp.expand((s[0, 0] - s[1, 1])/g**order), p1=sp.expand((p[0, 0] - p[1, 1])/e**order))
# material-function coefficients as linear forms in the constants c_term
C = {tm: sp.Symbol('c_' + '_'.join(map(str, tm[0])) + ('__' + '_'.join(''.join(map(str, t_)) for t_ in tm[1]) if tm[1] else '')) for tm in res}
eta4 = sum(C[tm]*r['s12'] for tm, r in res.items() if r['order'] == 5); psi2 = sum(C[tm]*r['n1'] for tm, r in res.items() if r['order'] == 4)
p4 = sum(C[tm]*r['p1'] for tm, r in res.items() if r['order'] == 5)
chk4 = sum(C[tm]*r['p1'] for tm, r in res.items() if r['order'] == 4); print('4th-order terms contribute to eta_P1:', sp.simplify(chk4), '(must be 0)')
# composite from general shear expansions
b1, eta2, E4, psi0, Psi2, eps = sp.symbols('b1 eta2 eta4 psi0 psi2 epsilon', real=True)
S = g*(b1 + eta2*g**2 + E4*g**4); N1 = g**2*(psi0 + Psi2*g**2); Dl = sp.sqrt(N1**2 + 4*S**2)
eeff = sp.series(g*S/Dl, g, 0, 6).removeO(); c3, c5 = sp.symbols('c3 c5'); gs = 2*eps + c3*eps**3 + c5*eps**5
eq = sp.expand(sp.series(eeff.subs(g, gs), eps, 0, 6).removeO() - eps); so = sp.solve([eq.coeff(eps, 3), eq.coeff(eps, 5)], [c3, c5], dict=True)[0]
comp = sp.expand(sp.series((Dl**2/(g*S)).subs(g, gs.subs(so)), eps, 0, 5).removeO())
k2 = sp.factor(comp.coeff(eps, 2)); k4 = sp.factor(sp.simplify(comp.coeff(eps, 4))); print('composite: eps^2 coeff =', k2, '; eps^4 coeff =', k4)
D4 = sp.expand(p4 - k4.subs({E4: eta4, Psi2: psi2}))
lin = {str(c): sp.factor(D4.coeff(c)) for c in C.values() if D4.coeff(c) != 0}
print('number of 4th/5th-order spanning terms:', sum(r['order'] == 4 for r in res.values()), sum(r['order'] == 5 for r in res.values()))
print('constants surviving in D4 with their coefficients:')
for k_, v in lin.items(): print('   ', k_, ':', v)
rest = sp.factor(sp.simplify(D4 - sum(sp.Symbol(k_)*v for k_, v in lin.items()))); print('remainder (lower-order constants):', rest)
json.dump(dict(lin={k_: str(v) for k_, v in lin.items()}, rest=str(rest), k4=str(k4)), open('oq_fifth.json', 'w'))
