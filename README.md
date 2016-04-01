sympy2fenics
============

Generate FEniCS expressions using vector calculus with sympy.

Dependency
----------

Only `sympy`.

Example
-------

    u = sympy.Matrix(sympy.sympify("""
        (sin(pi*x)*sin(pi*y), 
         x*(1.0-x)*y*(1.0-y))
        """))

    print("divergence of u:")
    print(sympy2exp(div(u)))

    print("symmetric gradient of u:")
    print(sympy2exp(sym(grad(u.transpose(), dim = 2))))
