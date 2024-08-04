import numpy as np
import matplotlib.pyplot as plt


def euler_method(f, x0, y0, x_end, h):

    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    while x < x_end:
        y = y + h * f(x, y)
        x = x + h
        x_values += x,
        y_values += y,
    
    return x_values, y_values


def runge_kutta_method(f, x0, y0, x_end, h):

    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    while x < x_end:
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)
        
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x = x + h
        x_values += x,
        y_values += y,
    
    return x_values, y_values



if __name__ == '__main__':

    # Differential equation function
    def f(x, y):
        return x + y

    def exact_solution(x):
        return -x - 1 + 2 * np.exp(x)

    x0 = 0
    y0 = 1
    x_end = 10
    h = 0.1

    x_exact = np.linspace(x0, x_end, 1000)
    y_exact = exact_solution(x_exact)

    x_values_euler, y_values_euler = euler_method(f, x0, y0, x_end, h)

    x_values_rk, y_values_rk = runge_kutta_method(f, x0, y0, x_end, h)

    plt.plot(x_values_euler, y_values_euler, label='Euler Method Approximation', linestyle='--')
    plt.plot(x_values_rk, y_values_rk, label='Runge-Kutta Method Approximation', linestyle='--')
    plt.plot(x_exact, y_exact, label='Exact Solution', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Euler and Runge-Kutta methods vs. Exact Solution')
    plt.legend()
    plt.show()



