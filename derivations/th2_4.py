"""Separable K-BKZ fluid, arbitrary memory function and damping function: slow-flow defect through O(eps^4), both projections."""
import sympy as sp, json
a_, b_, gam, eps, s = sp.symbols('a b gamma epsilon s', positive=True); M = sp.symbols('M1:8')
def mint(expr): expr = sp.expand(expr); return sum(expr.coeff(s, j)*M[j - 1] for j in range(1, 8))
hser = lambda I: 1 - a_*I + b_*I**2
tr = lambda e, n: sp.series(sp.expand(e), eps, 0, n).removeO()
Ssh = mint(hser(gam**2*s**2)*gam*s); Nsh = mint(hser(gam**2*s**2)*gam**2*s**2)
I1 = tr(4*sp.sinh(eps*s)**2, 6)
t1 = sp.expand(mint(tr(hser(I1)*tr(2*sp.sinh(2*eps*s), 7), 6))/eps)                 # (s11 - s22)/eps
t2 = sp.expand(mint(tr(hser(I1)*tr(1 - sp.exp(-2*eps*s), 7), 6))/eps)               # (s33 - s22)/eps  (Finger: C^-1_33 = 1)
Dl = sp.sqrt(Nsh**2 + 4*Ssh**2); eeff = sp.series(gam*Ssh/Dl, gam, 0, 6).removeO(); c3, c5 = sp.symbols('c3 c5'); gs = 2*eps + c3*eps**3 + c5*eps**5
eq = sp.expand(sp.series(eeff.subs(gam, gs), eps, 0, 6).removeO() - eps); so = sp.solve([eq.coeff(eps, 3), eq.coeff(eps, 5)], [c3, c5], dict=True)[0]
c1 = sp.series((Dl**2/(gam*Ssh)).subs(gam, gs.subs(so)), eps, 0, 5).removeO()
c2 = sp.series((Dl*(Dl - Nsh)/(2*gam*Ssh)).subs(gam, gs.subs(so)), eps, 0, 5).removeO()
eta0, mu1, mu2, mu3, mu4 = sp.symbols('eta0 mu1 mu2 mu3 mu4'); mus = [1, mu1, mu2, mu3, mu4, sp.Symbol('mu5'), sp.Symbol('mu6')]
tomu = {M[j - 1]: sp.factorial(j)*eta0*mus[j - 1] for j in range(1, 8)}
out = {}
for nm, tru, com in (('P1', t1, c1), ('P2', t2, c2)):
    diff = sp.expand(tru - com)
    for k in (0, 1, 2, 3, 4):
        ck = sp.factor(sp.simplify(diff.coeff(eps, k))); cm = sp.factor(sp.simplify(ck.subs(tomu)))
        print(f'  {nm}: eps^{k}: {ck}   ==  {sp.expand(cm)}'); out[f'{nm}_{k}'] = dict(M=str(ck), mu=str(sp.expand(cm)))
json.dump(out, open('theory2_4.json', 'w'))
