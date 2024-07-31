import numpy as np


def rectangle_integral(a, b, f, n=1000):
    h = (b - a) / n
    S = sum([f(a + i * h) for i in range(int(n))])
    return h * S


def trapezoidal_integral(a, b, f, n=1000):
    h = (b - a) / n
    S = f(a) + sum([2 * f(a + i * h) for i in range(1, int(n))]) + f(b)
    return (h / 2) * S 


def simpson_integral(a, b, f, n=1000):
    h = (b - a) / n
    S = f(a) + sum([4 * f(a + (i * h)) for i in range(1, int(n), 2)]) + sum([2 * f(a + (i * h)) for i in range(2, int(n), 2)]) + f(b)
    return (h / 3) * S


def monte_carlo_integral(lower_bounds, upper_bounds, f, n=1000):

    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    n = int(n)
    dim = len(lower_bounds)
    
    def generate_random():
        return lower_bounds + (upper_bounds - lower_bounds) * np.random.uniform(size=dim)
    
    def compute_hypervolume():
        return np.prod(upper_bounds - lower_bounds)
    
    total_sum = sum([f(generate_random()) for _ in range(n)])
    hypervolume = compute_hypervolume()
    
    return hypervolume * (total_sum / n)



if __name__ == '__main__':

    def f(x):
        return x * x * x * x
    
    a, b, n = 0, 1, 1e+5

    print(rectangle_integral(a, b, f, n))
    print(trapezoidal_integral(a, b, f, n))
    print(simpson_integral(a, b, f, n))

    lower_bounds = [a]
    upper_bounds = [b]
    print(monte_carlo_integral(lower_bounds, upper_bounds, f, n))