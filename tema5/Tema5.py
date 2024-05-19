import numpy as np
from scipy.linalg import eigh, norm, cholesky, svd, inv


def jacobi_method(A, epsilon=1e-10, max_iterations=1000):
    n = A.shape[0]
    U = np.identity(n)
    for _ in range(max_iterations):
        A_off_diagonal = A - np.diag(np.diagonal(A))
        largest_idx = np.unravel_index(np.argmax(np.abs(A_off_diagonal)), A.shape)
        p, q = largest_idx

        if A[p, q] == 0 or np.abs(A[p, q]) < epsilon:
            break

        if A[p, p] == A[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan2(2 * A[p, q], A[p, p] - A[q, q])

        c = np.cos(theta)
        s = np.sin(theta)


        for i in range(n):
            if i != p and i != q:
                aip, aiq = A[i, p], A[i, q]
                A[i, p] = A[p, i] = c * aip + s * aiq
                A[i, q] = A[q, i] = c * aiq - s * aip

        app, aqq, apq = A[p, p], A[q, q], A[p, q]
        A[p, p] = A[p, p] = c * c * app + 2 * c * s * apq + s * s * aqq
        A[q, q] = A[q, q] = s * s * app - 2 * c * s * apq + c * c * aqq
        A[p, q] = A[q, p] = 0.0

        for i in range(n):
            uip, uiq = U[i, p], U[i, q]
            U[i, p] = c * uip + s * uiq
            U[i, q] = -s * uip + c * uiq

    eigenvalues = np.diagonal(A)
    eigenvectors = U
    return eigenvalues, eigenvectors


def cholesky_sequence(A, epsilon=1e-10, max_iterations=100):
    for k in range(max_iterations):
        L = cholesky(A, lower=True)
        A_next = L.T @ L
        if norm(A - A_next) < epsilon:
            print("Converged after", k, "iterations")
            break
        A = A_next
    return A


def generate_symmetric_matrix(n, p):
    M = np.random.rand(p, n)
    A_symmetric = M + M.T  # symmetric
    return A_symmetric

def generate_matrix(n, p):
    A = np.random.rand(p, n)
    return A


def generate_positive_definite_matrix(n):
    M = np.random.rand(n, n)
    A_positive_definite = M @ M.T + n * np.eye(n)  # positive-definite
    return A_positive_definite


def svd_analysis(A, epsilon=1e-10):
    U, S, Vh = svd(A, full_matrices=False)
    S_inv = np.diag([1 / x if x > 1e-10 else 0 for x in S])

    # Extinde S_inv la dimensiunea corectă
    S_inv_expanded = np.zeros((Vh.shape[0], U.shape[1]))
    np.fill_diagonal(S_inv_expanded, [1 / x if x > epsilon else 0 for x in S])

    # Valorile singulare ale matricei A
    singular_values = S

    # Rangul matricei A
    rank_A = np.sum(S > 1e-10)
    rank_A_2 = np.linalg.matrix_rank(A)

    # Numărul de condiționare al matricei A
    condition_number = S[0] / S[-1] if rank_A > epsilon else np.inf
    condition_number_2 = np.linalg.cond(A)

    # Pseudoinversa Moore-Penrose a matricei A
    AI = Vh.T @ S_inv_expanded @ U.T

    # Pseudoinversa în sensul celor mai mici pătrate
    AJ = inv(A.T @ A) @ A.T

    # Norma dintre AI și AJ
    norm_difference = norm(AI - AJ, 1)

    return singular_values, rank_A, rank_A_2, condition_number, condition_number_2, AI, norm_difference


def main():
    p = int(input("Enter the dimension p: "))
    n = int(input("Enter the dimension n: "))

    if p<n:
        print("p must be greater than or equal to n")
    elif p==n:
        #A_symmetric = generate_symmetric_matrix(n, p)
        A_positive_definite = generate_positive_definite_matrix(n)
        A_init = np.copy(A_positive_definite)


        # Jacobi method for eigenvalues and eigenvectors
        eigenvalues, eigenvectors = jacobi_method(A_positive_definite)
        verification_norm = norm(A_init @ eigenvectors - eigenvectors @ np.diag(eigenvalues))

        # Cholesky decomposition sequence
        A_last = cholesky_sequence(A_positive_definite)

        print("Eigenvalues:", eigenvalues)
        print("Eigenvectors:\n", eigenvectors)
        print("Norma verificare:", verification_norm)
        print("Ultima matrice Cholensky:\n", A_last)


    elif p>n:
        A = generate_matrix(n, p)
        singular_values, rank_A, rank_A_2, condition_number, condition_number_2, AI, norm_difference = svd_analysis(A)
        print("Valorile singulare ale matricei A:", singular_values)
        print("Rangul matricei A:", rank_A)
        print("Rangul matricei A (2):", rank_A_2)
        print("Numărul de condiționare al matricei A:", condition_number)
        print("Numărul de condiționare al matricei A (2):", condition_number_2)
        print("Pseudoinversa Moore-Penrose a matricei A:\n", AI)
        print("Norma dintre pseudoinverse:", norm_difference)


if __name__ == "__main__":
    main()