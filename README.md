sympy2fenics
============

Scalar, vector, and matrix symbolic calculus using sympy with an interface
to generate FEniCS expressions.

By Lizao Li <lzlarryli@gmail.com>

Dependency
----------

Only `sympy`.

Example
-------

    from sympy2fenics import *

    f = str2sympy('sin(x)')                   # a scalar function in 1D
    u = str2sympy('(sin(x), sin(y))')         # a vector function in 2D
    w = str2sympy('((x,y),(x,z))')            # a matrix funciton in 2D
    v = str2sympy('sin(x)*sin(y)')            # a scalar function in 2D

    print("divergence of w:")
    print(Div(w))

    print("symmetric gradient of u:")
    print(Epsilon(u))

    print("symmetric gradient of u computed directly:")
    print(Sym(Grad(u.transpose())))
