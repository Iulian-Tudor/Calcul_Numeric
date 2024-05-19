import numpy as np
import math as math

#np.dot - produs de 2 array-uri
#np.fill_diagonal - umple diagonala unei matrici cu o valoare
#np.linalg.norm - norma matricii
#np.eye - matricea cu diagonala 1
#np.zeros - matricea de 0-uri

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


#decompozitia matricei in L(inferioara) si U(superioara)
def decomposition(A, epsilon):
    n = len(A)
    # Check if the matrix is square
    if A.shape[0] != A.shape[1]:
        print("The matrix is not square")
        return

    # Decompose the matrix A into L and U
    for k in range(n):

        for i in range(k, n):
            sum_L = 0
            for s in range(k):
                sum_L += A[i, s] * A[s, k]
            A[i, k] = A[i, k] - sum_L

        for j in range(k + 1, n):
            sum_U = 0
            for s in range(k):
                sum_U += A[k, s] * A[s, j]
            if(abs(A[k, k]) > epsilon): # Avoid division by zero
                A[k, j] = (A[k, j] - sum_U) / A[k, k]
            else:
                print("Sistem incompatibil sau compatibil nedeterminat")
                return None

    print("Test matrice A: ",A)

    print("Matricea L este: \n:", np.tril(A), "\n")
    print("Matricea U este: \n:", np.triu(A), "\n")

    return np.tril(A), np.triu(A)

#rezolvarea sistemului prin metoda substitutiei directe si inverse
def direct(L,b,epsilon):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        sum = np.dot(L[i, :i], y[:i])
        if abs(L[i,i]) < epsilon:
            print("Sistem incompatibil sau compatibil nedeterminat")
            return None
        else:
            y[i] = (b[i] - sum) / L[i,i]
    return y

def inversa(U,y, epsilon):
    n = len(y)
    x = np.zeros(n)
    for i in range(n):
        U[i,i] = 1
    for i in range(n-1,-1,-1):
        sum = np.dot(U[i, i+1:], x[i+1:])
        if abs(U[i,i]) < epsilon:
            print("Sistem incompatibil sau compatibil nedeterminat")
            return None
        else:
            x[i] = (y[i] - sum) / U[i,i] #ToDo
    return x

def solve(L,U,b, epsilon):
    y = direct(L,b,epsilon)
    x = inversa(U,y,epsilon)
    return x

#calculul determinantului
def determinant(L,U):
    return np.prod(np.diag(L))

def euclidean_norm(x):
    return np.linalg.norm(x, ord=2)

def main():

    print("Dati dimensiunea n: ")
    n = int(input())

    #precizia calculului
    epsilon = 1e-10
    max_cond = 10**3

    A = generate_well_conditioned_matrix(n)
    b = np.random.rand(n)

    A_init = A.copy()
    b_init = b.copy()

    print("Matricea A este:\n",A)
    print("\n")
    print("Inversa Matricii A este:\n",np.linalg.inv(A))
    print("\n")

    L, U = decomposition(A,epsilon)
    det_A = determinant(L,U)
    xLU = solve(L,U,b,epsilon)

    A_init = np.array(A_init)
    xLU = np.array(xLU)
    print("Determinantul matricei este: ",det_A)
    print(np.linalg.det(A_init))
    print("Solutia sistemului este: ",xLU)
    x_exact = np.linalg.solve(A_init, b_init)

    # subpunctul 4
    norm1 = euclidean_norm(A_init.dot(xLU) - b_init)

    # subpunctul 6
    norm2 = euclidean_norm(xLU-x_exact)
    norm3 = euclidean_norm(xLU - np.dot(np.linalg.inv(A_init), b_init))

    print("Norma Residuala: ",norm1)
    print("Norma || xLU - xlib ||: ",norm2)
    print("Norma || xLU - A_inv_lib * b_init ||: ",norm3)



if __name__ == "__main__":
    main()