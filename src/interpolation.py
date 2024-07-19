import numpy as np
from .estimators import least_squares
import matplotlib.pyplot as plt


class PolynomialInterpolation1D:

    def __init__(self, x, y, degree):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.degree = degree
        self.a = None
        self.X = np.vander(self.x, self.degree + 1, increasing=True)
        self.applied_method = None

    def least_squares_solver(self):
        self.a = least_squares(self.X, self.y)

    def solve(self):
        assert self.degree <= len(self.x) - 1, 'degree > n - 1'
        if self.degree == len(self.x) - 1:
            self.applied_method = "linear system"
            self.a = np.linalg.solve(self.X, self.y) 
            return
        self.applied_method = "least squares"
        self.least_squares_solver()
    
    def interpolate(self, x):
        x = np.asarray(x)
        fx = 0
        for i in range(len(self.a)):
            fx += self.a[i] * x ** i
        return fx        

    def plot(self, min_x=None, max_x=None, m=None):
        if min_x is None:
            min_x = 2 * min(self.x)
        if max_x is None:
            max_x = 2 * max(self.x)
        if m is None:
            m = 100
        plt.scatter(self.x, self.y)
        x = np.linspace(min_x, max_x, m)
        fx =  self.interpolate(x)
        plt.plot(x, fx)
        plt.show()



if __name__ == '__main__':
    x, y, deg = [-7, -3, 0, 3, 7], [5.1, 1.9, 1.1, 2.1, 4.9], 4
    interpolator = PolynomialInterpolation1D(x, y, deg)
    interpolator.solve()
    print(interpolator.applied_method)
    print('interpolated scalar', interpolator.interpolate(5))
    print('interpolated array', interpolator.interpolate(x))
    interpolator.plot()
