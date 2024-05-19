import math
import random
import numpy as np
import tkinter as tk
from tkinter import messagebox

#ex1
def smallest_u():
    m = 1
    u = 10 ** (-m)
    u_smallest = float('inf')

    while True:
        if (1 + u !=1) and (1 + u/10 ==1):
            u_smallest = u
            break
        m = m+1
        u = 10 ** (-m)

    return u_smallest

#ex2
def neassociative_sum(u):
    x = 1.0
    y = u/10
    z = u/10

    a = (x + y) + z
    b = x + (y + z)

    print("a = ", a)
    print("b = ", b)

    return a, b

#find 3 random numbers where multiplication is not associative
def neassociative_prod():
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    z = random.uniform(0, 1)

    a = (x * y) * z
    b = x * (y * z)

    print("a = ", a)
    print("b = ", b)

    if a != b:
        return a, b
    else:
        a = 0
        b = 0
        return a, b

#ex3
def T(i,a):
    match i:
        case 1:
            return a
        case 2:
            return 3*a / (3-a**2)
        case 3:
            return (15 * a - a ** 3) / (15 - 6 * a ** 2)
        case 4:
            return (105 * a - 10 * a ** 3) / (105 - 45 * a ** 2 + a ** 4)
        case 5:
            return (945 * a - 105 * a ** 3 + a ** 5) / (945 - 420 * a ** 2 + 15 * a ** 4)
        case 6:
            return (10395 * a - 1260 * a ** 3 + 21 * a ** 5) / (10395 - 4725 * a ** 2 + 210 * a ** 4 - a ** 6)
        case 7:
            return (135135*a - 17235*a**3 + 378*a**5 - a**7) / (135135 - 62370*a**2 + 3150*a**4 - 28*a**6)
        case 8:
            return (2027025 * a - 270270 * a ** 3 + 6930 * a ** 5 - 36 * a ** 7) / (
                2027025 - 945945 * a ** 2 + 51975 * a ** 4 - 630 * a ** 6 + a ** 8)
        case 9:
            return (34459425 * a - 4729725 * a ** 3 + 135135 * a ** 5 - 990 * a ** 7 + a ** 9) / (
                34459425 - 16216200 * a ** 2 + 945945 * a ** 4 - 13860 * a ** 6 + 45 * a ** 8)

def aproximare_tan():
    numere_random = [random.uniform(-math.pi/2, math.pi/2) for i in range(10000)]

    valori_t = {i: [T(i,a) for a in numere_random] for i in range (1,10)}

    eroare = {i: sum(abs(T(i,a) - math.tan(a)) for a in numere_random) / len(numere_random) for i in range(1,10)}

    top_3 = sorted(eroare, key=eroare.get)[:3]

    print("Top 3 cele mai bune aproximari sunt: ", top_3)
    print("Erorile asociate sunt: ", [eroare[i] for i in top_3])

    return top_3


#BONUS
def sin(n,a):
    T_na = T(n,a)
    return T_na / math.sqrt(1 + T_na ** 2)

def cos(n,a):
    T_na = T(n,a)
    return 1 / math.sqrt(1 + T_na ** 2)

def aproximare_sin_cos():
    numere_random = [random.uniform(-math.pi/2, math.pi/2) for i in range(10000)]

    valori_sin = {i: [sin(i,a) for a in numere_random] for i in range (1,10)}
    valori_cos = {i: [cos(i,a) for a in numere_random] for i in range (1,10)}

    eroare_sin = {i: sum(abs(sin(i,a) - math.sin(a)) for a in numere_random) / len(numere_random) for i in range(1,10)}
    eroare_cos = {i: sum(abs(cos(i,a) - math.cos(a)) for a in numere_random) / len(numere_random) for i in range(1,10)}

    top_3_sin = sorted(eroare_sin, key=eroare_sin.get)[:3]
    top_3_cos = sorted(eroare_cos, key=eroare_cos.get)[:3]

    print("Top 3 cele mai bune aproximari pentru sin sunt: ", top_3_sin)
    print("Erorile asociate sunt: ", [eroare_sin[i] for i in top_3_sin])

    print("Top 3 cele mai bune aproximari pentru cos sunt: ", top_3_cos)
    print("Erorile asociate sunt: ", [eroare_cos[i] for i in top_3_cos])

    return top_3_sin, top_3_cos

def clear_screen(text_widget):
    text_widget.delete('1.0', tk.END)

def execute_function(func, text_widget):
    match func:
        case "Smallest u":
            clear_screen(text_widget)
            result = smallest_u()
            text_widget.insert(tk.END, f"The smallest u is: {result}\n")
        case "Neassociative Sum":
            clear_screen(text_widget)
            u = smallest_u()
            a, b = neassociative_sum(u)
            text_widget.insert(tk.END, f"a = {a}, b = {b}\n")
            if a == b:
                text_widget.insert(tk.END, "Asociative Sum\n")
            else:
                text_widget.insert(tk.END, "Neasociative Sum\n")
        case "Neassociative Product":
            clear_screen(text_widget)
            a, b = neassociative_prod()
            text_widget.insert(tk.END, f"a = {a}, b = {b}\n")
            if a == b:
                text_widget.insert(tk.END, "Asociative Multiplication\n")
            else:
                text_widget.insert(tk.END, "Neasociative Multiplication\n")
        case "Approximate tan":
            clear_screen(text_widget)
            result = aproximare_tan()
            text_widget.insert(tk.END, f"Top 3 approximations for tan: {result}\n")
        case "Approximate sin and cos":
            clear_screen(text_widget)
            sin_result, cos_result = aproximare_sin_cos()
            text_widget.insert(tk.END, f"Top 3 approximations for sin: {sin_result}\nTop 3 approximations for cos: {cos_result}\n")

def create_gui():
    root = tk.Tk()
    root.title("Function Executor")

    text_widget = tk.Text(root, height=20, width=50)
    text_widget.grid(row=0, column=0, columnspan=3)

    buttons = [
        "Smallest u",
        "Neassociative Sum",
        "Neassociative Product",
        "Approximate tan",
        "Approximate sin and cos",
        "Clear Screen"
    ]

    row = 1
    col = 0
    for button_text in buttons:
        if button_text == "Clear Screen":
            button = tk.Button(root, text=button_text, command=lambda: clear_screen(text_widget))
        else:
            button = tk.Button(root, text=button_text, command=lambda b=button_text: execute_function(b, text_widget))
        button.grid(row=row, column=col, sticky="nsew")
        col += 1
        if col == 3:
            col = 0
            row += 1

    # Configure row and column weights
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    for i in range(1, row+1):
        root.grid_rowconfigure(i, weight=1)
    for i in range(3):
        root.grid_columnconfigure(i, weight=1)

    root.mainloop()

if __name__ == "__main__":
    create_gui()