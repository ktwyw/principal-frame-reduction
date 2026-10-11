import sympy as sp
g,e=sp.symbols('g e')
Z=sp.zeros(3)
A_s=sp.Matrix([[0,g,0],[g,0,0],[0,0,0]]); B_s=sp.Matrix([[-2*g**2,0,0],[0,0,0],[0,0,0]]); C_s=Z; D_s=Z
A_p=sp.diag(2*e,-2*e,0); B_p=sp.diag(-4*e**2,-4*e**2,0); C_p=sp.diag(8*e**3,-8*e**3,0); D_p=sp.diag(-16*e**4,-16*e**4,0)
tr=lambda M: M.trace()
def terms(A,B,C,D):
    s=lambda X,Y: X*Y+Y*X
    return {
     'trA2^2 A': tr(A*A)**2*A, 'trA3 A2': tr(A**3)*A*A,
     'A2B+BA2': s(A*A,B), 'ABA': A*B*A, 'trA2 (AB+BA)': tr(A*A)*s(A,B), 'trAB A2': tr(A*B)*A*A,
     'trA2B A': tr(A*A*B)*A, 'trA3 B': tr(A**3)*B, 'trA2 trAB? (weight5) A trAB*trA..':Z*0+tr(A*A)*tr(A*B)*sp.eye(3)*0,
     'AB2+B2A': s(A,B*B), 'BAB': B*A*B, 'trB2 A': tr(B*B)*A, 'trAB B': tr(A*B)*B, 'trAB2? ': Z,
     'AD+DA': s(A,D), 'BC+CB': s(B,C), 'A2C+CA2': s(A*A,C), 'ACA': A*C*A, 'trA2 C': tr(A*A)*C, 'trAC A': tr(A*C)*A,
    }
ts=terms(A_s,B_s,C_s,D_s); tp=terms(A_p,B_p,C_p,D_p)
for k in ts:
    eta4=sp.simplify(ts[k][0,1]/g**5); p4=sp.simplify((tp[k][0,0]-tp[k][1,1])/e**5)
    psi=sp.simplify((ts[k][0,0]-ts[k][1,1])/g**4)
    print(f"{k:28s} eta4={eta4!s:6s} Psi1(g^2 coef)={psi!s:6s} p4={p4!s:6s}  p4-64eta4={sp.simplify(p4-64*eta4)}")
