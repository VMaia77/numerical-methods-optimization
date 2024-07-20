import numpy as np
import autograd as ad


def bisection_method(f, x_neg, x_pos, eps=1e-15):
    x = None
    while abs(x_pos - x_neg) > eps:
        x = (x_pos + x_neg) / 2

        if f(x) > 0:
            x_pos = x
        else:
            x_neg = x
    return x


def newton_raphson_method(f, x, df=None, n=10, eps=1e-7):

    if df is None:
        gradient_f = ad.grad(f)
        def df(x):
            return gradient_f(x)
    
    counter = 0
    diff = float('inf') 
    
    while counter < n and diff > eps:
        counter += 1
        x_new = x - f(x) / df(x)
        diff = np.abs(x_new - x)
        x = x_new
    
    return x



if __name__ == '__main__':

    def f(x):
        return x ** 2

    init_x_neg = -2
    init_x_pos = 2
    print(bisection_method(f, init_x_neg, init_x_pos))

    def df(x):
        return 2 * x
    
    print(newton_raphson_method(f, x=-2.0, df=df, n=100, eps=1e-15))
    print(newton_raphson_method(f, x=-2.0, df=None, n=100, eps=1e-15))