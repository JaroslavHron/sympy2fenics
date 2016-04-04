import sympy

def sympy2exp(exp):
    x, y, z = sympy.symbols('x[0] x[1] x[2]')    
    def to_ccode(f):
        f = f.subs('x', x).subs('y', y).subs('z', z)
        raw = sympy.printing.ccode(f)
        return raw.replace("M_PI", "pi")
    if hasattr(exp, "__getitem__"):
        if exp.shape[0] == 1 or exp.shape[1] == 1:
            return tuple(map(to_ccode, exp))
        else:
            return tuple([tuple(map(to_ccode, exp[i, :]))
                          for i in range(exp.shape[1])])
    else:
        return to_ccode(exp)

def infer_dim(u):
    atoms = u.atoms()
    if sympy.sympify('z') in atoms:
        return 3
    elif sympy.sympify('y') in atoms:
        return 2
    else:
        return 1    
    
def grad(u, dim = None):
    if not dim:
        dim = infer_dim(u)
    # transpose first if it is a row vector
    if u.is_Matrix and u.shape[0] != 1:
        u = u.transpose()
    # take the gradient
    if dim == 1:
        return sympy.Matrix([u.diff('x')]).transpose()
    elif dim == 2:
        return sympy.Matrix([u.diff('x'), u.diff('y')]).transpose()
    elif dim == 3:
        return sympy.Matrix(
            [u.diff('x'), u.diff('y'), u.diff('z')]).transpose()

def curl(u):
    if u.is_Matrix and min(u.args) == 1:
        # 3D vector curl
        return sympy.Matrix([u[2].diff('y') - u[1].diff('z'),
                             u[0].diff('z') - u[2].diff('x'),
                             u[1].diff('x') - u[0].diff('y')])
    else:
        # 2D rotated gradient
        return sympy.Matrix([u.diff('y'), -u.diff('x')])

def rot(u):
    # 2d rot
    return u[1].diff('x') - u[0].diff('y')

def div(u):
    def vec_div(w):
        if w.shape[0] == 2:
            return w[0].diff('x') + w[1].diff('y')
        elif u.shape[0] == 3:
            return w[0].diff('x') + w[1].diff('y') + w[2].diff('z')
    if u.shape[1] == 1:
        # vector
        return vec_div(u)
    else:
        # matrix
        result = []
        for i in range(u.shape[1]):
            result.append(vec_div(u.row(i).transpose()))
        return sympy.Matrix(result)
    
def sym(u):
    return (u + u.transpose()) / 2.0

def tr(u):
    return u.trace()

def hess(u, dim = None):
    return grad(grad(u, dim), dim)


if __name__ == '__main__':
    u2 = sympy.sympify('x*y')
    v2 = sympy.Matrix(sympy.sympify('(x*y, y*x)'))
    w2 = sympy.Matrix(sympy.sympify('((x*y, sin(y)), (cos(x), pi*y^2))'))
    u3 = sympy.sympify('a*x*y*z')
    v3 = sympy.Matrix(sympy.sympify('(x*y, y*x, x*y*z)'))
    w3 = sympy.Matrix(sympy.sympify(
        """
        ((x*y,    sin(y), z), 
         (cos(x), pi*y^2, y),
         (z*x,    x*y*z,  x))
        """))
