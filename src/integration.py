import numpy as np

class MonteCarloIntegration:

    def __init__(self, lower_bounds, upper_bounds, n=1000):
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.n = int(n)
        self.dim = len(lower_bounds)

    def integrate(self):
        total_sum = sum([f(self.generate_random()) for _ in range(self.n)])
        hypervolume = self.compute_hypervolume()
        return hypervolume * total_sum / self.n

    def compute_hypervolume(self):
        return np.prod(self.upper_bounds - self.lower_bounds)

    def generate_random(self):
        return self.lower_bounds + (self.upper_bounds - self.lower_bounds) * np.random.uniform(size=self.dim)


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


if __name__ == '__main__':

    def f(x):
        return x * x
    
    a, b, n = 0, 1, 1e+5

    print(rectangle_integral(a, b, f, n))
    print(trapezoidal_integral(a, b, f, n))
    print(simpson_integral(a, b, f, n))

    lower_bounds = [a]
    upper_bounds = [b]
    algorithm = MonteCarloIntegration(lower_bounds, upper_bounds, n)
    print(algorithm.integrate())
