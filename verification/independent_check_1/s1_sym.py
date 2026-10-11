# Symbolic checks: Sec II, App A retarded motion, uniaxial SOF
import sympy as sp
g,e=sp.symbols('gd ed',positive=True)
s12,N1,N2=sp.symbols('s12 N1 N2',positive=True)
# --- Sec II
chi=sp.atan2(2*s12,N1)/2
Ds=sp.sqrt(N1**2+4*s12**2)
D=sp.Matrix([[0,g/2,0],[g/2,0,0],[0,0,0]])
c,s=sp.cos(chi),sp.sin(chi)
Q=sp.Matrix([[c,-s,0],[s,c,0],[0,0,1]])
Dp=sp.simplify(Q.T*D*Q)
eeff=g*s12/Ds
print('D11 - eeff',sp.simplify(Dp[0,0]-eeff),' D12 - h', sp.simplify(Dp[0,1]-g/2*N1/Ds), 'D22+eeff', sp.simplify(Dp[1,1]+eeff))
h=g/2*N1/Ds
print('II:',sp.simplify(8*(eeff**2+h**2)-2*g**2), sp.simplify(8*eeff**2/(2*g**2)-1/(1+(N1/(2*s12))**2)))
P1=Ds/eeff
print('P1 form2:',sp.simplify(P1-4*(s12/g)*(1+(N1/(2*s12))**2)))
print('Trouton:',sp.simplify(P1/(s12/g)-(g/eeff)**2))
# principal stresses, with s33=0 reference: s22 = N2, s11 = N1+N2
S=sp.Matrix([[N1+N2,s12,0],[s12,N2,0],[0,0,0]])
sm=(S[0,0]+S[1,1])/2-Ds/2; spl=(S[0,0]+S[1,1])/2+Ds/2
print('eig check', sp.simplify((S-sm*sp.eye(3)).det()), sp.simplify((S-spl*sp.eye(3)).det()))
P2=(0-sm)/eeff
print('P2:',sp.simplify(P2-Ds*(Ds-N1-2*N2)/(2*g*s12)))
# stress frame check: Q^T S Q diagonal
print('offdiag in eigenframe', sp.simplify((Q.T*S*Q)[0,1]))

# --- App A: retarded motion
b1,b2,b11,b3,b12,b111=sp.symbols('b1 b2 b11 b3 b12 b111')
def rme(L):
    g1=L+L.T
    g2=-L*g1-g1*L.T
    g3=-L*g2-g2*L.T
    dd=sum(g1[i,j]*g1[j,i] for i in range(3) for j in range(3))
    sig=b1*g1+b2*g2+b11*g1*g1+b3*g3+b12*(g1*g2+g2*g1)+b111*dd*g1
    return g1,g2,g3,sp.expand(sig)
L=sp.zeros(3,3); L[0,1]=g
g1,g2,g3,sg=rme(L)
print('shear g2',g2.tolist(),'g3',g3.tolist())
S12=sg[0,1]; n1=sg[0,0]-sg[1,1]; n2=sg[1,1]-sg[2,2]
print('s12',sp.factor(S12),'N1',n1,'N2',n2)
L=sp.diag(e,-e,0)
g1,g2,g3,sgp=rme(L)
print('planar g1',g1.tolist(),'g2',g2.tolist(),'g3',g3.tolist())
eP1=sp.expand((sgp[0,0]-sgp[1,1])/e); eP2=sp.expand((sgp[2,2]-sgp[1,1])/e)
print('etaP1',sp.collect(eP1,e)); print('etaP2',sp.collect(eP2,e))
# composites
Dsg=sp.sqrt(n1**2+4*S12**2)
ee=sp.series(g*S12/Dsg,g,0,6).removeO()
# invert
x=sp.symbols('x',positive=True)
c1,c3,c5=sp.symbols('c1 c3 c5')
ginv=c1*x+c3*x**3
eq=sp.expand(ee.subs(g,ginv)-x)
sol1=sp.solve(eq.coeff(x,1),c1)[0]
sol3=sp.solve(eq.coeff(x,3).subs(c1,sol1),c3)[0]
ginv=ginv.subs({c1:sol1,c3:sol3})
cP1=sp.series((Dsg/(g*S12/Dsg)).subs(g,ginv),x,0,4).removeO()
cP2=sp.series((Dsg*(Dsg-n1-2*n2)/(2*g*S12)).subs(g,ginv),x,0,3).removeO()
print('[P1]',sp.collect(sp.expand(cP1),x))
print('P1-[P1]',sp.factor(sp.expand(eP1.subs(e,x)-cP1)))
d2=sp.expand(eP2.subs(e,x)-cP2)
print('P2-[P2] (through x^2)',[sp.factor(d2.coeff(x,k)) for k in range(3)])
# uniaxial SOF
L=sp.diag(e,-e/2,-e/2)
_,_,_,su=rme(L)
etaE=sp.expand((su[0,0]-su[1,1])/e)
print('etaE to O(e):', etaE.coeff(e,0), sp.factor(etaE.coeff(e,1)), ' claimed 3/2(Psi1+2Psi2)=',sp.expand(sp.Rational(3,2)*(-2*b2+2*b11)))
