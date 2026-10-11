import sympy as sp
g,e,b1,e2,e4,P0,p2c,p2,p4=sp.symbols('g e b1 eta2 eta4 Psi10 psi2 p2 p4')
eta=b1+e2*g**2+e4*g**4; Psi=P0+p2c*g**2
s12=eta*g; N1=Psi*g**2
r2=(N1/(2*s12))**2
comp=sp.series(4*eta*(1+r2),g,0,6).removeO()
ee=sp.series(g/2/sp.sqrt(1+r2),g,0,7).removeO()
# invert ee(g)=e
G=sp.symbols('G1:6'); gs=2*e+G[1]*e**3+G[3]*e**5
eqs=sp.expand(sp.series(ee.subs(g,gs)-e,e,0,7).removeO())
sol=sp.solve([eqs.coeff(e,3),eqs.coeff(e,5)],[G[1],G[3]],dict=True)[0]
gs=gs.subs(sol)
compe=sp.expand(sp.series(comp.subs(g,gs),e,0,6).removeO())
diff=sp.expand(4*b1+p2*e**2+p4*e**4-compe)
print('e^2:',sp.simplify(diff.coeff(e,2)))
print('e^4:',sp.simplify(diff.coeff(e,4)))
claim2=p2-16*e2-4*P0**2/b1; claim4=p4-64*e4-32*P0*p2c/b1-4*P0**4/b1**3
print(sp.simplify(diff.coeff(e,2)-claim2), sp.simplify(diff.coeff(e,4)-claim4))
