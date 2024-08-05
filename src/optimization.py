import numpy as np
from autograd import grad
from matplotlib import pyplot as plt


def gradient_descent(f, bounds, n, alpha, momentum=0.0, df=None, verbose=True):

    if df is None:
        df = grad(f)

    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    
    x_values = []
    y_values = []
    
    v = np.zeros_like(x)
    
    for i in range(n):

        g = df(x)
        
        v = momentum * v + alpha * g
        
        x = x - v
        
        x_values += x.copy(),
        y_values += f(x),
        
        if verbose:
            print(f'Gradient Descent Iteration {i+1}: x = {x}, function value = {f(x)}')
    
    return np.array(x_values), np.array(y_values)


def adam(f, bounds, n, alpha, beta1, beta2, epsilon=1e-8, df=None, verbose=True):

    if df is None:
        df = grad(f)

    dim = bounds.shape[0]

    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    m = np.zeros(dim)
    v = np.zeros(dim)

    x_values = []
    y_values = []

    for t in range(1, n + 1):

        g = df(x)
        
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g ** 2)
        
        m_corrected = m / (1.0 - beta1 ** t)
        v_corrected = v / (1.0 - beta2 ** t)
        
        x = x - alpha * m_corrected / (np.sqrt(v_corrected) + epsilon)
        
        x_values += x.copy(),
        y_values += f(x),
        
        if verbose:
            print(f'Adam Iteration {t}: x = {x}, function value = {f(x)}')

    return np.array(x_values), np.array(y_values)



def plot_optimization(dim, bounds, f, gd_solutions, gd_scores, adam_solutions, adam_scores):

    if dim == 1:
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Generate inputs and compute function values
        inputs = np.linspace(bounds[0, 0], bounds[0, 1], 100)
        function_values = f(inputs)  # No need to reshape for 1D
        
        # Plot function and optimization paths
        ax.plot(inputs, function_values, label='f(x)')
        ax.plot(gd_solutions, gd_scores, '.-', color='green', label='Gradient Descent Path')
        ax.plot(adam_solutions, adam_scores, '.-', color='red', label='Adam Path')
        ax.set_title('Optimization in 1D')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()

    elif dim == 2:
        fig, ax = plt.subplots(figsize=(14, 6))

        # Generate grid for function plotting
        x = np.linspace(bounds[0, 0], bounds[0, 1], 100)
        y = np.linspace(bounds[1, 0], bounds[1, 1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute function values for the grid
        Z = f(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
        # Plot contours and optimization paths
        cs = ax.contourf(X, Y, Z, cmap='viridis', levels=50)
        fig.colorbar(cs, ax=ax)
        ax.plot(gd_solutions[:, 0], gd_solutions[:, 1], '.-', color='green', label='Gradient Descent Path')
        ax.plot(adam_solutions[:, 0], adam_solutions[:, 1], '.-', color='red', label='Adam Path')
        ax.set_title('Optimization Trajectories in 2D')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    
    import autograd.numpy as anp

    # def f(x):
    #     return np.sum(x ** 2, axis=-1)

    # def df(x):
    #     return np.array([2*x[0], 2*x[1]])

    # to use autograd
    def f(x):
        return anp.sum(x ** 2, axis=-1)
        
    df = None

    bounds = np.array([[-1, 1], [-1, 1]])  # Bounds for each dimension
    n = 50  # Number of iterations
    
    # Gradient Descent
    alpha_gd = 0.1
    momentum = 0.3
    gd_solutions, gd_scores = gradient_descent(f, bounds, n, alpha_gd, momentum, df)
    
    # Adam Optimization
    alpha_adam = 0.1
    beta1 = 0.5
    beta2 = 0.999
    epsilon = 1e-15
    adam_solutions, adam_scores = adam(f, bounds, n, alpha_adam, beta1, beta2, epsilon, df)
    
    plot_optimization(dim=2, 
                      bounds=bounds, 
                      f=f, 
                      gd_solutions=gd_solutions, 
                      gd_scores=gd_scores, 
                      adam_solutions=adam_solutions, 
                      adam_scores=adam_scores)
    
