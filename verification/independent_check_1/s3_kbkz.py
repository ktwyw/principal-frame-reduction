import sympy as sp
g,x,a,b=sp.symbols('g x a b',positive=True)
M=sp.symbols('M0:10',positive=True)
u=sp.symbols('u')
# shear
S12=M[1]*g-a*M[3]*g**3+b*M[5]*g**5
N1=M[2]*g**2-a*M[4]*g**4+b*M[6]*g**6
def tr(ex,var,n):
    return sp.series(ex,var,0,n+1).removeO()
Ds=sp.sqrt(N1**2+4*S12**2)
ee=tr(g*S12/Ds,g,5)
c1,c3,c5=sp.symbols('c1 c3 c5')
gi=c1*x+c3*x**3+c5*x**5
eq=sp.expand(ee.subs(g,gi)-x)
s1=sp.solve(eq.coeff(x,1),c1)[0]
s3=sp.solve(eq.coeff(x,3).subs(c1,s1),c3)[0]
s5=sp.solve(eq.coeff(x,5).subs({c1:s1,c3:s3}),c5)[0]
gi=gi.subs({c1:s1,c3:s3,c5:s5})
P1c=tr((Ds**2/(g*S12)).subs(g,gi),x,4)
P2c=tr((Ds*(Ds-N1)/(2*g*S12)).subs(g,gi),x,4)
# planar: integrand in u = eps*s
I=4*sp.sinh(u)**2
h=1-a*I+b*I**2
def mom(ex,n):
    ex=sp.expand(sp.series(ex,u,0,n+1).removeO())
    return sum(ex.coeff(u,k)*M[k]*x**k for k in range(n+1))
P1=sp.expand(mom(h*2*sp.sinh(2*u),5)/x)
P2=sp.expand(mom(h*(1-sp.exp(-2*u)),5)/x)
d1=sp.expand(P1-P1c); d2=sp.expand(P2-P2c)
print('P1 diff coefficients:')
for k in range(5): print(k, sp.factor(sp.simplify(d1.coeff(x,k))))
claim=4*(2*M[1]*M[3]-3*M[2]**2)/(3*M[1])*x**2+sp.Rational(4,15)*(2*M[5]-15*M[2]**4/M[1]**3-60*a*(M[5]-2*M[2]*M[4]/M[1]))*x**4
print('P1 diff - claim =',sp.simplify(d1-claim))
print('P2 diff coefficients:')
for k in range(5): print(k, sp.factor(sp.simplify(d2.coeff(x,k))))
# in terms of moments
eta0=sp.symbols('eta0',positive=True); t=sp.symbols('t1:6',positive=True)
sub={M[1]:eta0}
for j in range(2,7): sub[M[j]]=sp.factorial(j)*eta0*t[j-2]
for k in (2,3,4): print('P2 order',k, sp.factor(sp.simplify(d2.coeff(x,k).subs(sub)/eta0)))
for k in (2,4): print('P1 order',k, sp.expand(sp.simplify(d1.coeff(x,k).subs(sub)/eta0)))
