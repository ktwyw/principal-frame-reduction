# Protocol short-time series for multimode UCM, symbolic (3 modes)
import sympy as sp
t,e0=sp.symbols('t e0',positive=True)
nm=2
G=sp.symbols('G1:%d'%(nm+1),positive=True); r=sp.symbols('r1:%d'%(nm+1),positive=True)
NO=6
gs=sp.symbols('g0:%d'%NO)
gd=sum(gs[n]*t**n for n in range(NO))
def solve_lin(src, rate):
    # y' = src - rate*y, y(0)=0, power series
    y=0
    co=[0]*(NO+2)
    srcp=sp.Poly(sp.expand(src),t)
    sc=lambda n: srcp.coeff_monomial(t**n)
    for n in range(NO+1):
        co[n+1]=sp.expand((sc(n)-rate*co[n])/(n+1))
    return sum(co[n]*t**n for n in range(NO+2))
def trunc(ex,n):
    ex=sp.expand(ex)
    return sum(ex.coeff(t,k)*t**k for k in range(n+1))
S12=0;N1=0
for k in range(nm):
    s=trunc(solve_lin(G[k]*gd,r[k]),NO)
    n=trunc(solve_lin(trunc(2*gd*s,NO),r[k]),NO)
    S12+=s;N1+=n
# constraint gd^2 S12^2 = e0^2 (N1^2+4 S12^2)
con=sp.expand(gd**2*S12**2-e0**2*(N1**2+4*S12**2))
sol={}
# lowest order t^2: g0^2 = 4 e0^2
sol[gs[0]]=2*e0
for n in range(1,5):
    c=con.coeff(t,n+2).subs(sol)
    c=sp.expand(c)
    ss=sp.solve(c,gs[n])
    sol[gs[n]]=sp.factor(ss[0])
Gs=sum(G); m1=sum(G[k]*r[k] for k in range(nm))/Gs; m2=sum(G[k]*r[k]**2 for k in range(nm))/Gs
print('g1',sol[gs[1]]); print('g2',sp.simplify(sol[gs[2]]-e0**3)); 
print('g3 - claimed',sp.simplify(sol[gs[3]]-(e0**3/3)*(-m1)))
print('g4', sp.factor(sp.simplify(sol[gs[4]])))
S12s=trunc(S12.subs(sol),6); N1s=trunc(N1.subs(sol),6)
Ds2=trunc(N1s**2+4*S12s**2,7)   # starts at t^2
# planar
P=0
for k in range(nm):
    # p' = 2e(p+G) - r p ; q' = -2e(q+G) - r q
    def lin(a,src):
        co=[0]*(NO+2)
        co[1]=src
        for n in range(1,NO+1): co[n+1]=sp.expand(a*co[n]/(n+1))
        return sum(co[n]*t**n for n in range(NO+2))
    p=lin(2*e0-r[k],2*e0*G[k]); q=lin(-2*e0-r[k],-2*e0*G[k])
    P+=p-q
P=trunc(P,6)

A=sp.expand(Ds2/t**2); B=sp.expand(P/t)
# ratio R = sqrt(A)/B ; write sqrt(A)= sum a_n t^n by matching, then divide
NN=5
a=sp.symbols('a0:%d'%NN)
sq=sum(a[n]*t**n for n in range(NN))
eq=sp.expand(sq**2-A)
so={}
so[a[0]]=sp.sqrt(sp.factor(A.coeff(t,0)))
print('a0',so[a[0]])
for n in range(1,NN):
    c=eq.coeff(t,n).subs(so)
    so[a[n]]=sp.factor(sp.solve(c,a[n])[0])
sq=sq.subs(so)
# R*B = sq
R=sp.symbols('R0:%d'%NN); Rs=sum(R[n]*t**n for n in range(NN))
eq=sp.expand(Rs*B-sq); sr={}
for n in range(NN):
    c=eq.coeff(t,n).subs(sr)
    sr[R[n]]=sp.factor(sp.solve(c,R[n])[0])
    print('ratio coeff t^%d:'%n, sr[R[n]])
print('t^4 coeff + e0^2 Var/120 =',sp.simplify(sr[R[4]]+e0**2*(m2-m1**2)/120))
