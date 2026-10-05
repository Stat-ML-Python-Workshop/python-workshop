"""Week 4: tiny dense vectors and rectangular matrices."""
def dot(x, y):
    if len(x) != len(y):
        raise ValueError("vector dimensions differ")
    total = 0.0
    # BEGIN STUDENT: dot product
    raise NotImplementedError("Complete this week’s core exercise")
    # END STUDENT
    return total

def shape(matrix):
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("ragged matrix")
    return len(matrix), width

def matvec(matrix, vector):
    _, width = shape(matrix)
    if width != len(vector):
        raise ValueError("matrix/vector dimensions differ")
    return [dot(row, vector) for row in matrix]

def matmul(a, b):
    """Teacher demonstration/extension; not a week-4 required test."""
    _, inner = shape(a)
    rows, _ = shape(b)
    if inner != rows:
        raise ValueError("matrix dimensions differ")
    return [[dot(row, list(col)) for col in zip(*b)] for row in a]
