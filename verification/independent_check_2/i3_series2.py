import sympy as sp
a,t=sp.symbols('alpha t')
M=7
c=sp.symbols('c1:%d'%M); cs=[1]+list(c)
def B(m):
    return sp.factorial(m)/sp.prod([(i-a) for i in range(1,m+2)])
S=sum(2*cs[n]*B(n)*t**n for n in range(M))          # S / t^{1-a}
gam_t=sum(2*cs[n]*t**(n+1)/(n+1) for n in range(M))
r=sp.Symbol('r')
gdr=sum(2*cs[n]*r**n for n in range(M)); gamr=sum(2*cs[n]*r**(n+1)/(n+1) for n in range(M))
poly=sp.Poly(sp.expand(gdr*gam_t-gdr*gamr),r)
N=sum(2*coef*B(m)*t**m for (m,),coef in poly.terms())  # N/t^{1-a}
gd=sum(2*cs[n]*t**n for n in range(M))
eq=sp.expand(S**2*(gd**2-4)-N**2)
sol={}
for p in range(0,8):
    cp=sp.expand(eq.coeff(t,p).subs(sol))
    if cp==0: continue
    free=[ci for ci in c if cp.has(ci) and ci not in sol]
    if not free:
        print('order',p,'residual',sp.simplify(cp)); continue
    k=min(free,key=lambda s:c.index(s))
    sol[k]=sp.factor(sp.solve(cp,k)[0]); print(k,'=',sol[k])
Dp=4*sum(2**(2*j)/sp.factorial(2*j)*t**(2*j)/(2*j+1-a) for j in range(4))
num=sp.expand((gd*S).subs(sol))
ratio=sp.series(num/Dp,t,0,5).removeO()
for p in range(5): print('ratio t^%d:'%p, sp.factor(sp.simplify(ratio.coeff(t,p))))
print(sp.simplify(ratio.coeff(t,2).subs(a,sp.Rational(1,2))))
