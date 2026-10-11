import sympy as sp
a,t=sp.symbols('alpha t',positive=True)
M=7
c=sp.symbols('c1:%d'%M)
cs=[1]+list(c)
B=lambda m: sp.gamma(1-a)*sp.gamma(m+1)/sp.gamma(m+2-a)
# gd(s)=2*sum cs[n] s^n ; gamma(s)=2 sum cs[n] s^{n+1}/(n+1)
S=sum(2*cs[n]*B(n)*t**(n+1-a) for n in range(M))
gam_t=sum(2*cs[n]*t**(n+1)/(n+1) for n in range(M))
# N=2 int u^-a [gd(t-u)gam(t) - gd(t-u)gam(t-u)] du ; poly in r=t-u
r=sp.Symbol('r')
gdr=sum(2*cs[n]*r**n for n in range(M)); gamr=sum(2*cs[n]*r**(n+1)/(n+1) for n in range(M))
poly=sp.Poly(sp.expand(gdr*gam_t-gdr*gamr),r)
N=0
for (m,),coef in poly.terms():
    N+=2*coef*B(m)*t**(m+1-a)
gd=sum(2*cs[n]*t**n for n in range(M))
# protocol: S^2 (gd^2-4) - N^2 =0 ; divide by t^{2-2a}
eq=sp.expand((S**2*(gd**2-4)-N**2)/t**(2-2*a))
sol={}
for order in range(1,M):
    co=sp.expand(eq).coeff(t,order+1)  # leading in gd^2-4 ~ t^order -> eq ~ t^{order}... find
for k in range(1,M):
    # collect coefficient of t^(k) after substituting known
    e=sp.expand(eq.subs(sol))
    # lowest power containing c_k
    for p in range(0,2*M):
        cp=e.coeff(t,p)
        if cp!=0 and cp.has(c[k-1]):
            s=sp.solve(cp,c[k-1]); sol[c[k-1]]=sp.simplify(s[0]); break
        elif cp!=0 and sp.simplify(cp)!=0 and not any(cp.has(ci) for ci in c[k-1:]):
            pass
    if k>=5: break
for k in range(1,5): print('c%d ='%k, sp.factor(sol[c[k-1]]))
e=sp.expand(eq.subs(sol))
print('residual low orders:',[sp.simplify(e.coeff(t,p)) for p in range(0,7)])
Dp=4*sum(2**(2*j)/sp.factorial(2*j)*t**(2*j+1-a)/(2*j+1-a) for j in range(4))
ratio=sp.expand(gd*S).subs(sol)/Dp
rs=sp.series(sp.simplify(ratio*1),t,0,6).removeO()
for p in range(0,6): print(p, sp.factor(sp.simplify(rs.coeff(t,p))))
print('alpha=1/2 x^2 coef', sp.nsimplify(sp.simplify(rs.coeff(t,2).subs(a,sp.Rational(1,2)))))
