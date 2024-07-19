import numpy as np


def gaussian_elimination_matrix_vector_solver(A, b):

    A_copy = np.copy(A)
    x = np.copy(b)
    n = len(x)
    
    # Forward elimination
    for k in range(n - 1):
        for i in range(k + 1, n):
            if A_copy[i, k] != 0.0:
                factor = A_copy[i, k] / A_copy[k, k]
                A_copy[i, k:n] = A_copy[i, k:n] - factor * A_copy[k, k:n]
                x[i] = x[i] - factor * x[k]

    # Back substitution
    for k in range(n - 1, -1, -1):
        x[k] = (x[k] - np.dot(A_copy[k, k + 1:n], x[k + 1:n])) / A_copy[k, k]

    return x


def gaussian_elimination_solver(A, B):

    B_was_1d = False
    if B.ndim == 1:
        B = B.reshape(-1, 1)
        B_was_1d = True

    n = len(A)
    augmented_matrix = np.hstack((A, B))
    
    # Forward elimination to form an upper triangular matrix
    for k in range(n):
        # Partial pivoting
        max_row_index = np.argmax(np.abs(augmented_matrix[k:, k])) + k
        if max_row_index != k:
            augmented_matrix[[k, max_row_index]] = augmented_matrix[[max_row_index, k]]
        
        # Make the pivot element 1 by dividing the row by the pivot element
        pivot = augmented_matrix[k, k]
        if pivot == 0:
            raise ValueError("Matrix is singular and cannot be processed.")
        
        augmented_matrix[k] = augmented_matrix[k] / pivot
        
        # Eliminate the elements below the pivot
        for i in range(k + 1, n):
            factor = augmented_matrix[i, k]
            augmented_matrix[i] = augmented_matrix[i] - factor * augmented_matrix[k]
    
    # Back substitution to form an identity matrix on the left side
    for k in range(n - 1, -1, -1):
        for i in range(k - 1, -1, -1):
            factor = augmented_matrix[i, k]
            augmented_matrix[i] = augmented_matrix[i] - factor * augmented_matrix[k]
    
    # Extract the right half of the augmented matrix
    result = augmented_matrix[:, n:]

    if B_was_1d:
        result = result.flatten()

    return result


def gaussian_elimination_inversion(A):
    n = len(A)
    identity_matrix = np.eye(n)
    return gaussian_elimination_solver(A, identity_matrix)



if __name__ == '__main__':

    A = np.array([[1., 1.], 
                  [0.035, 0.05]])
    
    b = np.array([24000., 930.])

    print('matrix vector solution: ', gaussian_elimination_solver(A, b))
    print('matrix vector solver solution: ', gaussian_elimination_matrix_vector_solver(A, b))
    print('matrix vector np solution: ', np.linalg.solve(A, b))

    print('matrix matrix solution: ', gaussian_elimination_solver(A, A))
    print('matrix matrix np solution: ', np.linalg.solve(A, A))

    print('inverse: ', gaussian_elimination_inversion(A))
    print('np inverse', np.linalg.inv(A))

