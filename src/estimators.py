import numpy as np


def least_squares(X, y):
    ls_b = X.T @ y
    ls_X = X.T @ X
    return np.linalg.solve(ls_X, ls_b)