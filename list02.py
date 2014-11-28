from pprint import PrettyPrinter
from math import sqrt
import numpy as np
from numpy import linalg as LA
from pprint import pprint
from matplotlib.pylab import *

def cholesky(A):
    # Source: http://www.quantstart.com/articles/Cholesky-Decomposition-in-Python-and-NumPy
    """Performs a Cholesky decomposition of A, which must
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L."""
    n = len(A)

    # Create zero matrix for L
    L = [[0.0] * n for i in xrange(n)]

    # Perform the Cholesky decomposition
    for i in xrange(n):
        for k in xrange(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in xrange(k))

            if (i == k): # Diagonal elements
                # LaTeX: l_{kk} = \sqrt{ a_{kk} - \sum^{k-1}_{j=1} l^2_{kj}}
                L[i][k] = sqrt(A[i][i] - tmp_sum)
            else:
                # LaTeX: l_{ik} = \frac{1}{l_{kk}} \left( a_{ik} - \sum^{k-1}_{j=1} l_{ij} l_{kj} \right)
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L

#A = [[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]]
#L = cholesky(A)
#
#A = [[1, 4], [4, 1]]
#print(cholesky(A))


def solve_equations(hessian_value, gradient_value):
    try:
        L = cholesky(hessian_value)
        hessian_inverse = LA.inv(hessian_value)
        dk = np.dot(hessian_inverse, gradient_value)
    except:
        dk = -gradient_value
    return L, dk

def linear_search_backtracking(function, current_x, dk, gdk, function_value, c1):
    wolfe1 = False
    alpha = 1
    while not wolfe1:
        next_x = current_x + alpha * dk
        l1 = function(next_x)
        r1 = function(current_x) + c1 * alpha * gdk
        wolfe1 = l1 <= r1
        if not wolfe1:
            alpha = 0.5*alpha
    #wolfe2 = False
    #beta = 0.1
    #while not wolfe2:
    #    l2 = np.dot(np.transpose(gradient(next_x)), dk)
    #    r2 = c2 * gradient_dk
    #    wolfe2 = l2 >= r2
    return alpha, next_x, l1

def newton_solve(function, x0, gradient, hessian, **kwargs):
    gtolerance = kwargs.get('gtolerance', 1.0e-10)
    stolerance = kwargs.get('stolerance', 1.0e-10)
    max_iterations = kwargs.get('max_iterations', 100)
    tolerance2 = kwargs.get('theta', 0.9)
    c1 = kwargs.get('c1', 1.0e-4)
    c2 = kwargs.get('c2', 1.0e-4)
    #dk = kwargs.get('dk', None)

    if not (c1 <= c2 <= 1):
        raise Exception('Precondition: c1 <= c2 <= 1')

    current_x = x0
    snorm = 2*stolerance
    iteration = 0
    continue_iterating = True

    data = [['Iteration', 'f(x_k)', '||gf(x_k)||', '||dk||', 'alpha', 'x[0]', 'x[1]']]
    print('{0: ^10}\t{0: ^10}\t{0: ^10}\t{0: ^10}\t{0: ^10}\t{0: ^10}\t{0: ^10}'.format(
        'Iteration', 'f(x_k)', '||gf(x_k)||', '||dk||', 'alpha', 'x[0]', 'x[1]'
    ))

    while continue_iterating:
        function_value = function(current_x)
        gradient_value = gradient(current_x)
        hessian_value = hessian(current_x)
        gnorm = LA.norm(gradient_value)
        continue_iterating = not(gnorm < gtolerance)
        if not continue_iterating:
            return current_x
        # Solve the equation Ax=b equivalent to Hessian*x = -gradient_value
        L, dk = solve_equations(hessian_value, -gradient_value)

        gdk = np.dot(gradient_value, np.transpose(dk))
        snorm = math.sqrt(np.dot(dk, dk))

        function_value_temp = function_value
        # Check Armijo (first Wolfe condition)
        alpha, current_x, function_value = linear_search_backtracking(function, current_x, dk, gdk, function_value, c1)

        # Check second Wolfe condition
        alpha_min = 0.1 * stolerance / snorm

        data.append([iteration, function_value_temp, gnorm, snorm, alpha, current_x[0], current_x[1]])
        print('{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}'.format(
            iteration, function_value_temp, gnorm, snorm, alpha, current_x[0], current_x[1]
        ))

        g = gradient(current_x)
        gnorm = math.sqrt(np.dot(g, g))

        iteration += 1

        continue_iterating = not(gnorm < gtolerance or alpha*snorm < stolerance) and \
            iteration <= max_iterations
    return current_x, data

def rosen_function(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosen_gradient(x):
    return np.array((
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ))

def rosen_hessian(x):
    return np.array((
        (2 - 400*x[1] + 1200*x[0]**2,   -400*x[0]),
        (-400*x[0],                     200)
    ))

def plot(function):
    x = arange(-1.5, 1.5, 0.01)
    y = arange(-0.5, 1.5, 0.01)
    [X,Y] = meshgrid(x, y)
    Z = function(X, Y)
    contour(Z, x=X, y=Y, levels = 50)
    show()

#x0 = np.transpose([-2, -1])
#x0 = np.transpose([0, -3])
x0 = np.transpose([-2, 3])
dk=None
x, _ = newton_solve(rosen_function, x0, rosen_gradient, rosen_hessian)
print('Resultado = {}'.format(x))
