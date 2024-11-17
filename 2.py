import numpy as sp

t = sp.symbols('t')
A = sp.symbols('A', real=True, positive=True)

integrand = A**2 * sp.exp(-2*t)

result = sp.integrate(integrand, (t, 0, sp.oo))

print(result)
