import numpy as np
import numpy as np
import tkinter as tk
from tkinter import messagebox

def generate_well_conditioned_matrix(n):
    D = np.diagflat([1 + i/(n-1) for i in range(n)])
    H = np.random.rand(n, n)
    Q, _ = np.linalg.qr(H)
    A = Q @ D @ Q.T
    return A

def householder(A, epsilon):
    m, n = A.shape
    Q = np.eye(m) # Matricea ortogonală
    R = A.copy() # Matricea superior triunghiulară

    for j in range(n):
        # Creăm reflectorul Householder
        x = R[j:, j]
        normx = np.linalg.norm(x)
        rho = -np.sign(x[0])
        u1 = x[0] - rho * normx
        if abs(u1) < epsilon:
            print("Elementul de pe diagonala principală este prea mic pentru a continua.")
        else:
            u = x / u1

        beta = -rho * u1 / normx

        # Aplicăm transformarea Householder
        R[j:, :] = R[j:, :] - beta * np.outer(u, u).dot(R[j:, :])
        Q[:, j:] = Q[:, j:] - Q[:, j:].dot(beta * np.outer(u, u))

    return Q, R


def calculate_errors(x_QR, x_Householder, A, b, s, epsilon):
    if np.linalg.norm(s) < epsilon:
        print("Norma vectorului s este prea mică pentru a calcula erorile relative.")
        return None
    errors = {
        "||x_QR - x_Householder||": np.linalg.norm(x_QR - x_Householder),
        "||A*x_Householder - b||": np.linalg.norm(A.dot(x_Householder) - b),
        "||A*x_QR - b||": np.linalg.norm(A.dot(x_QR) - b),
        "||x_Householder - s|| / ||s||": np.linalg.norm(x_Householder - s)/np.linalg.norm(s),
        "||(x_QR-s)/s ||":np.linalg.norm((x_QR-s))/np.linalg.norm(s)
    }
    return errors

def calculate_limit(A, epsilon):
    diff = np.inf
    while diff > epsilon:
        Q, R = np.linalg.qr(A)
        A_next = np.zeros_like(A)
        R_upper = np.triu(R)  # Get the upper triangle of R

        # Iterate over the upper triangle of R, starting from the diagonal
        for i in range(R.shape[0]):
            for j in range(i, R.shape[1]):  # Only consider upper triangle elements
                for k in range(Q.shape[1]):
                    A_next[i, k] += R_upper[i, j] * Q[j, k]

        diff = np.linalg.norm(A_next - A)
        A = A_next
    return A


def main():
    n = int(input("Introduceți dimensiunea matricei n: "))
    epsilon = 1e-12
    A = generate_well_conditioned_matrix(n)
    s = np.random.rand(n)
    b = A.dot(s)
    Q, R = np.linalg.qr(A, 'complete')
    Qh, Rh = householder(A,epsilon)
    if abs(np.linalg.det(R)) < epsilon:
        print("Determinantul matricei R este prea mic pentru a calcula inversa.")
        return
    x_QR = np.linalg.solve(R, Q.T.dot(b))
    x_Householder = np.linalg.solve(Rh, Qh.T.dot(b))
    errors = calculate_errors(x_QR, x_Householder, A, b, s, epsilon)
    if errors is not None:
        for error_name, error_value in errors.items():
            if error_value > 1e-6:
                print("Aceasta eroare nu este destul de mică: ", error_name, error_value)
            print(f"{error_name}: {error_value}")
    inv_A_qr_based=np.dot(np.linalg.inv(R), Q.T)
    inv_A_lib=np.linalg.inv(A)
    print("Norma diferenței între inversa bazată pe qr și inversa funcției de bibliotecă: ",np.linalg.norm(inv_A_qr_based-inv_A_lib))

    print("\n")

    A_limit = calculate_limit(A, epsilon)
    print("Matricea limită este:")
    print(A_limit)

if __name__ == "__main__":
    main()
