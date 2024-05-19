import numpy as np
import math as math

#generarea unei matrici bine conditionate
def generate_well_conditioned_matrix(n):
    # Generăm o matrice diagonală cu valori proprii între 1 și 2
    D = np.diagflat([1 + i/(n-1) for i in range(n)])
    # Generăm o matrice ortogonală Q de dimensiune n x n
    H = np.random.rand(n, n)
    Q, _ = np.linalg.qr(H)
    # Calculăm matricea A = QDQ^T care este bine condiționată
    A = Q @ D @ Q.T
    return A

def lu_decomposition_bonus(A, epsilon):
    n = len(A)
    L = np.zeros((n * (n + 1)) // 2)
    U = np.zeros((n * (n + 1)) // 2)

    for i in range(n):
        for k in range(i, n):
            # Calculam suma pentru L prin formula L[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(i))
            total_L = sum(L[i * (i + 1) // 2 + j] * U[j * (j + 1) // 2 + k - j] for j in range(i))
            U[i * (i + 1) // 2 + k - i] = A[i][k] - total_L

        for k in range(i, n):
            if (i == k):
                L[i * (i + 1) // 2 + i] = 1
            else:
                total_U = sum(L[k * (k + 1) // 2 + j] * U[j * (j + 1) // 2 + i - j] for j in range(i))
                if abs(U[i * (i + 1) // 2]) < epsilon:
                    print("Sistem incompatibil sau compatibil nedeterminat")
                    return None
                else:
                    L[k * (k + 1) // 2 + i] = (A[k][i] - total_U) / U[i * (i + 1) // 2]

    return L, U


def solve_lu_bonus(L, U, b, epsilon):
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - sum(L[i * (i + 1) // 2 + j] * y[j] for j in range(i))

    for i in range(n - 1, -1, -1):
        if abs(U[i * (i + 1) // 2]) < epsilon:
            print("Sistem incompatibil sau compatibil nedeterminat")
            return None
        else:
            x[i] = (y[i] - sum(U[i * (i + 1) // 2 + j - i] * x[j] for j in range(i + 1, n))) / U[i * (i + 1) // 2]

    return x

def main():
    n=int(input("Dati dimensiunea n: "))

    epsilon = 1e-12
    max_cond = 10**3

    A = generate_well_conditioned_matrix(n)
    b = np.random.rand(n)

 #Bonus
    Lb, Ub = lu_decomposition_bonus(A, epsilon)
    x = solve_lu_bonus(Lb, Ub, b, epsilon)
    print("Solutia la bonus:", x)

if __name__ == "__main__":
    main()