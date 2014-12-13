import math

try:
    import numpy as np
    from numpy import linalg as LA
except:
    print("Install numpy: sudo aptitude install python-numpy")

try:
    from matplotlib.pylab import *
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt
except:
    print("(Optional) Install matplotlib: sudo aptitude install python-matplotlib")


def cholesky(A):
    """
    Performs a Cholesky decomposition of A, which must
    be a symmetric and positive definite matrix. The function
    returns the lower variant triangular matrix, L.
    This code snippet has been taken from [1].
    [1] http://www.quantstart.com/articles/Cholesky-Decomposition-in-Python-and-NumPy
    Tests:
    A = [[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]]
    L = cholesky(A)

    A = [[1, 4], [4, 1]]
    print(cholesky(A))
    This should give an error
    """
    n = len(A)

    # Create zero matrix for L
    L = [[0.0] * n for i in xrange(n)]

    # Perform the Cholesky decomposition
    for i in xrange(n):
        for k in xrange(i+1):
            tmp_sum = sum(L[i][j] * L[k][j] for j in xrange(k))

            if (i == k): # Diagonal elements
                # LaTeX: l_{kk} = \sqrt{ a_{kk} - \sum^{k-1}_{j=1} l^2_{kj}}
                L[i][k] = math.sqrt(A[i][i] - tmp_sum)
            else:
                # LaTeX: l_{ik} = \frac{1}{l_{kk}} \left( a_{ik} - \sum^{k-1}_{j=1} l_{ij} l_{kj} \right)
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))
    return L



def solve_equations(hessian_value, gradient_value):
    """
    Solves the equation system given by the hessian and the gradient.
    It returns the dk (direction) vector
    """
    try:
        L = cholesky(hessian_value)
        hessian_inverse = LA.inv(hessian_value)
        dk = np.dot(hessian_inverse, gradient_value)
    except:
        dk = -gradient_value
    return dk


def linear_search_backtracking(function, current_x, dk, gdk, function_value, c1):
    """
    Perform a linear search.
    It returns the alpha, x and f(x)
    """
    wolfe1 = False
    alpha = 1
    while not wolfe1:
        next_x = current_x + alpha * dk
        l1 = function(next_x)
        r1 = function(current_x) + c1 * alpha * gdk
        wolfe1 = l1 <= r1
        if not wolfe1:
            alpha = 0.5*alpha
    return alpha, next_x, l1


def newton_solve(function, x0, gradient, hessian, **kwargs):
    """
    Newton line search method.
    It returns x* the optimum value and the history of the iterations
    """
    gtolerance = kwargs.get('gtolerance', 1.0e-10)
    stolerance = kwargs.get('stolerance', 1.0e-10)
    max_iterations = kwargs.get('max_iterations', 100)
    c1 = kwargs.get('c1', 1.0e-4)
    c2 = kwargs.get('c2', 1.0e-4)

    if not (c1 <= c2 <= 1):
        raise Exception('Precondition: c1 <= c2 <= 1')

    current_x = x0
    snorm = 2*stolerance
    iteration = 0
    continue_iterating = True

    data = [['Iteration', 'f(x_k)', '||gf(x_k)||', '||dk||', 'alpha', 'x[0]', 'x[1]']]
    print('{0: ^10}\t{1: ^10}\t{2: ^10}\t{3: ^10}\t{4: ^10}\t{5: ^10}\t{6: ^10}'.format(
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
        dk = solve_equations(hessian_value, -gradient_value)

        gdk = np.dot(gradient_value, np.transpose(dk))
        snorm = math.sqrt(np.dot(dk, dk))

        function_value_temp = function_value
        # Check Armijo (first Wolfe condition)
        alpha, current_x, function_value = linear_search_backtracking(function, current_x, dk, gdk, function_value, c1)

        # Check second Wolfe condition
        alpha_min = 0.1 * stolerance / snorm

        data.append([iteration, function_value_temp, gnorm, snorm, alpha, current_x[0], current_x[1]])
        print('{:15.8f}\t{:15.8f}\t{:15.8f}\t{:15.8f}\t{:15.8f}\t{:15.8f}\t{:15.8f}'.format(
            iteration, function_value_temp, gnorm, snorm, alpha, current_x[0], current_x[1]
        ))

        g = gradient(current_x)
        gnorm = math.sqrt(np.dot(g, g))

        iteration += 1

        continue_iterating = not(gnorm < gtolerance or alpha*snorm < stolerance) and \
            iteration <= max_iterations
    return current_x, data


def plot_function(function):
    """
    Plots the function.
    All the functions works for many variables except for this one that only works
    for two variables. An interface mapping function is needed. I could not
    evaluate the meshgrid function using first class functions with arrays =(
    """
    fig = plt.figure()
    #ax = Axes3D(fig, azim = -128, elev = 43)
    #ax = Axes3D(fig, azim = 115, elev =45)
    ax = Axes3D(fig, azim = 115, elev =90)

    s = .05
    X = np.arange(-2, 2.+s, s) #arange(start,finish,increment), stores resulting vector in X
    Y = np.arange(-1, 3.+s, s)
    X, Y = np.meshgrid(X, Y)   #create the mesh grid
    Z = map(function, X, Y)
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = cm.jet)
    CS = plt.contour(X, Y, Z)  #plot contour
    plt.clabel(CS,inline=1, fontsize=10)
    CB = plt.colorbar(CS, shrink=0.8, extend='both')  #colorbar
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_contour(function):
    """
    I tried to show a 2D contour plot but it does not work properly, this function can be ignored.
    It will only remain here as a proof that I tried.
    """
    #x = arange(-1.5, 1.5, 0.01)
    #y = arange(-0.5, 1.5, 0.01)
    x = arange(-100, 250, 50)
    y = arange(-100, 250, 50)
    [X, Y] = meshgrid(x, y)
    Z = function(X, Y)
#    XY = map(zip, x, y)
#    Z = map(function, XY)
#    import ipdb; ipdb.set_trace() # BREAKPOINT
    plt.figure()
    im = plt.imshow(Z, interpolation='bilinear', origin='lower',
                    cmap=cm.gray, extent=(-3,3,-2,2))
    levels = np.arange(-1.2, 1.6, 0.2)
    CS = plt.contour(Z, levels,
                    origin='lower',
                    linewidths=2,
                    extent=(-3,3,-2,2))

    #Thicken the zero contour.
    zc = CS.collections[6]
    plt.setp(zc, linewidth=4)

    plt.clabel(CS, levels[1::2],  # label every second level
            inline=1,
            fmt='%1.1f',
            fontsize=14)

    # make a colorbar for the contour lines
    CB = plt.colorbar(CS, shrink=0.8, extend='both')

    plt.title('Lines with colorbar')
    #plt.hot()  # Now change the colormap for the contour lines and colorbar
    plt.flag()

    # We can still add a colorbar for the image, too.
    CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)

    # This makes the original colorbar look a bit out of place,
    # so let's improve its position.

    l,b,w,h = plt.gca().get_position().bounds
    ll,bb,ww,hh = CB.ax.get_position().bounds
    CB.ax.set_position([ll, b+0.1*h, ww, h*0.8])

    plt.show()

    #contour(Z, x=X, y=Y, levels = 50)
    #show()



###
### TEST CASE 1: ROSENBROCK FUNCTION
###

def rosen_function(x):
    """
    The testing function: The classic Rosenbrock function[1] has been chosen,
    [1]: https://en.wikipedia.org/wiki/Rosenbrock_function
    """
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2


def rosen_gradient(x):
    """
    The gradient of the testing function
    """
    return np.array((
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ))


def rosen_hessian(x):
    """
    The hessian of the testing function
    """
    return np.array((
        (2 - 400*x[1] + 1200*x[0]**2,   -400*x[0]),
        (-400*x[0],                     200)
    ))


def rosen_function_interface(x, y):
    """
    This is an interface for the evaluation of the function,
    This function is only used for ploting purposes,
    unfortunately I could not do this using the rosen_function directly
    """
    return rosen_function([x, y])


#x0 = np.transpose([-2, -1])
#x0 = np.transpose([0, -3])
x0 = np.transpose([-2, 3])
dk=None
x, _ = newton_solve(rosen_function, x0, rosen_gradient, rosen_hessian)
print('Solution = {}'.format(x))
try:
    plot_function(rosen_function_interface)
    #plot_contour(rosen_function_interface)
except:
    print('Matplotlib not installed, can not plot')


###
### TEST CASE 2: ANOTHER CONVEX FUNCTION
###
#def convex_function(x):
#    return 0.875*x[0]**2 + 0.8*x[1]**2 -350*x[0] - 300*x[1]
#
#def convex_gradient(x):
#    return np.array((
#        1.75*x[0] - 350,
#        1.6*x[1] - 300
#    ))
#
#def convex_hessian(x):
#    return np.array((
#        (1.75,   0),
#        (0,    1.6)
#    ))
#
#x0 = np.transpose([0, 0])
#x, _ = newton_solve(convex_function, x0, convex_gradient, convex_hessian)
#print('Solution = {}'.format(x))
##plot(convex_function)
