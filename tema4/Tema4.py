import numpy as np

def read_sparse_matrix_list_of_lists(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    n = int(lines[0])
    sparse_matrix = [[(i, 0.0)] for i in range(n)]
    for line in lines[1:]:
        parts = line.split(',')
        value, i, j = float(parts[0]), int(parts[1]), int(parts[2])
        found = False
        for index, (col, val) in enumerate(sparse_matrix[i]):
            if col == j:
                if i == j and val + value == 0:
                    raise ValueError(f"Diagonal element at row {i + 1} sums to zero.")
                if val != 0.0:  # Check if the position already had a non-zero value
                    print(f"Updating value at position ({i}, {j}) from {val} to {val + value}")
                sparse_matrix[i][index] = (col, val + value)
                found = True
                break
        if not found:
            #print(f"Adding new element at position ({i}, {j}) with value {value}")  # New element being added
            sparse_matrix[i].append((j, value))
    return sparse_matrix

def read_sparse_matrix_crs(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    n = int(lines[0])
    values = []
    col_indices = []
    row_pointers = [0]

    for line in lines[1:]:
        parts = line.split(',')
        value, i, j = float(parts[0]), int(parts[1]), int(parts[2])

        # Append value and column index
        values.append(value)
        col_indices.append(j)

        # row_pointers is extended to the correct length if skipping rows
        while len(row_pointers) <= i:
            row_pointers.append(len(values) - 1)

    # adding the total number of elements as the last element
    row_pointers.append(len(values))

    return values, col_indices, row_pointers

def read_vector(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    vector_b = [float(line.strip()) for line in lines[1:]]
    return vector_b

def check_diagonal_nonzero(sparse_matrix):
    for i, row in enumerate(sparse_matrix):
        diagonal_element = next((value for j, value in row if i == j), None)
        if diagonal_element is None:
            print(f"Lipsește elementul de pe diagonală la rândul {i}.")
            return False
        elif diagonal_element == 0:
            print(f"Elementul de pe diagonală la rândul {i} este zero.")
            return False
    return True


def gauss_seidel(sparse_matrix, b, epsilon=1e-10, max_iterations=1000):
    n = len(b)
    x = [0.0 for _ in range(n)]
    for iteration in range(max_iterations):
        max_error = 0  #maximum error for this iteration
        for i in range(n):
            sum_a = sum(value * x[j] for j, value in sparse_matrix[i] if j != i)
            new_xi = (b[i] - sum_a) / next(value for j, value in sparse_matrix[i] if j == i)
            max_error = max(max_error, abs(x[i] - new_xi))  # Update the maximum error
            x[i] = new_xi  # Update x[i] in-place
        if max_error > 1e+30:
            print("Divergenta", max_error)
            return x, iteration
        if max_error <= epsilon:
            return x, iteration
    return x, max_iterations

def gauss_seidel_crs(values, col_indices, row_pointers, b, epsilon=1e-10, max_iterations=1000):
    n = len(b)
    x = [0.0 for _ in range(n)]
    for iteration in range(max_iterations):
        max_error = 0
        for i in range(n):
            row_start = row_pointers[i]
            row_end = row_pointers[i + 1]
            sum_a = 0.0
            diag_val = None
            for idx in range(row_start, row_end):
                col_idx = col_indices[idx]
                if col_idx == i:
                    diag_val = values[idx]
                else:
                    sum_a += values[idx] * x[col_idx]
            if diag_val is None:
                raise ValueError("Diagonal element missing or zero")
            new_xi = (b[i] - sum_a) / diag_val
            max_error = max(max_error, abs(x[i] - new_xi))
            x[i] = new_xi
        if max_error > 1e+30:
            print("Divergenta", max_error)
            return x, iteration
        if max_error <= epsilon:
            return x, iteration
    return x, max_iterations


def calculate_residual(sparse_matrix, x, b):
    n = len(x)
    residual = [0.0 for _ in range(n)]
    for i in range(n):
        sum_a = sum(value * x[j] for j, value in sparse_matrix[i])
        residual[i] = abs(sum_a - b[i])
    return max(residual)

def calculate_residual_crs(values, col_indices, row_pointers, x, b):
    n = len(x)
    residual = [0.0 for _ in range(n)]
    for i in range(n):
        row_start = row_pointers[i]
        row_end = row_pointers[i + 1]
        row_sum = sum(values[idx] * x[col_indices[idx]] for idx in range(row_start, row_end))
        residual[i] = abs(row_sum - b[i])
    return max(residual)

def add_sparse_matrices(matrix_a, matrix_b):
    matrix_sum = [[(i, 0.0)] for i in range(len(matrix_a))]

    for i in range(len(matrix_a)):
        row_elements = {col: val for col, val in matrix_a[i][1:]}
        for col, val in matrix_b[i][1:]:
            row_elements[col] = row_elements.get(col, 0.0) + val

        matrix_sum[i].extend(sorted(row_elements.items()))

    return matrix_sum

def matrices_equal(matrix_sum, matrix_c, epsilon=1e-10):
    for i in range(len(matrix_sum)):
        sum_elements = {col: val for col, val in matrix_sum[i][1:]}
        c_elements = {col: val for col, val in matrix_c[i][1:]}

        all_cols = set(sum_elements.keys()) | set(c_elements.keys())
        for col in all_cols:
            val_sum = sum_elements.get(col, 0.0)
            val_c = c_elements.get(col, 0.0)
            if abs(val_sum - val_c) > epsilon:
                return False

    return True

def main():

    matrix_file = 'a_1.txt'
    vector_file = 'b_1.txt'

    sparse_matrix = read_sparse_matrix_list_of_lists(matrix_file)
    sparse_matrix_CRS= read_sparse_matrix_crs(matrix_file)
    vector_b = read_vector(vector_file)

    if not check_diagonal_nonzero(sparse_matrix):
        print("Elementele de pe diagonală ale matricei conțin zerouri. Gauss-Seidel nu poate fi aplicat.")
        return

    print("Sparse Matrix in list of lists: ")
    x, iterations = gauss_seidel(sparse_matrix, vector_b)
    print(f"Soluția sistemului este: {x[1:10]}")
    print(f"Numărul de iterații: {iterations}")

    residual = calculate_residual(sparse_matrix, x, vector_b)
    print(f"Rezidualul soluției este: {residual}")

    print("\n")

    values, col_indices, row_pointers = read_sparse_matrix_crs(matrix_file)
    vector_b = read_vector(vector_file)

    print("Sparse Matrix in CRS format: ")
    x, iterations = gauss_seidel_crs(values, col_indices, row_pointers, vector_b)
    print(f"Soluția sistemului este: {x[1:10]}")
    print(f"Numărul de iterații: {iterations}")

    # Calculate the residual
    residual = calculate_residual_crs(values, col_indices, row_pointers, x, vector_b)
    print(f"Rezidualul soluției este: {residual}")

    print("\n")

    print("Bonus\n")

    matrix_a = read_sparse_matrix_list_of_lists('a.txt')
    matrix_b = read_sparse_matrix_list_of_lists('b.txt')
    matrix_sum = add_sparse_matrices(matrix_a, matrix_b)

    matrix_c = read_sparse_matrix_list_of_lists('aplusb.txt')
    is_equal = matrices_equal(matrix_sum, matrix_c)

    print("Suma matricelor este egală cu matricea din aplusb.txt:", is_equal)


if __name__ == "__main__":
    main()
