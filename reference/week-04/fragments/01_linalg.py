"""Week 4: shapes, transposition, and matrix multiplication."""

def shape(matrix):
    """Return (rows, columns) for a nonempty rectangular matrix."""
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be nonempty")
    rows = len(matrix)
    columns = len(matrix[0])
    for row in matrix:
        if len(row) != columns:
            raise ValueError("ragged matrix")
    return rows, columns

def transpose(matrix):
    """Return a new matrix whose rows are the input columns."""
    rows, columns = shape(matrix)
    result = []
    for i in range(columns):
        result_row = []
        for j in range(rows):
            result_row.append(matrix[j][i])
        result.append(result_row)
    return result

def matmul(m1, m2):
    """Return the product of compatible nonempty rectangular matrices."""
    rows_m1, columns_m1 = shape(m1)
    rows_m2, columns_m2 = shape(m2)
    if columns_m1 != rows_m2:
        raise ValueError("m1 and m2 dimensions are incompatible")
    result = [[0.0 for j in range(columns_m2)] for i in range(rows_m1)]
    for i in range(rows_m1):
        for j in range(columns_m2):
            total = 0.0
            for k in range(columns_m1):
                total += m1[i][k] * m2[k][j]
            result[i][j] = total
    return result
