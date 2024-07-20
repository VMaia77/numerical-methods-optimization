


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