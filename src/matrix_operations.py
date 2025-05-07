"""
Matrix operations library with a bug in sparse matrix multiplication.
"""

import numpy as np
from scipy import sparse


def matrix_multiply(A, B):
    """
    Multiply two matrices A and B.
    
    Parameters:
    -----------
    A : numpy.ndarray or scipy.sparse matrix
        First matrix
    B : numpy.ndarray or scipy.sparse matrix
        Second matrix
        
    Returns:
    --------
    numpy.ndarray or scipy.sparse matrix
        Result of matrix multiplication
    """
    # Verify matrix dimensions for multiplication
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Matrix dimensions don't match for multiplication: {A.shape} and {B.shape}")
    
    # Bug: The function doesn't properly handle mixed sparse and dense matrices
    if sparse.issparse(A) and sparse.issparse(B):
        # Both sparse - this works fine
        return A @ B
    elif not sparse.issparse(A) and not sparse.issparse(B):
        # Both dense - this works fine
        return A @ B
    else:
        # Mixed case - this is where the bug occurs
        # The dimensions are checked correctly, but the operation fails
        # because of incompatible matrix types
        return A @ B  # This will raise a TypeError or similar


def test_matrix_multiply():
    """
    Test function to demonstrate the bug in matrix_multiply.
    """
    # Create test matrices
    A_dense = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
    B_dense = np.array([[7, 8], [9, 10], [11, 12]])  # 3x2 matrix
    
    # Convert to sparse format
    A_sparse = sparse.csr_matrix(A_dense)
    B_sparse = sparse.csr_matrix(B_dense)
    
    # Both dense - should work
    print("Testing dense * dense:")
    result1 = matrix_multiply(A_dense, B_dense)
    print(result1)
    print()
    
    # Both sparse - should work
    print("Testing sparse * sparse:")
    result2 = matrix_multiply(A_sparse, B_sparse)
    print(result2.toarray())
    print()
    
    # Mixed case 1 - will fail
    print("Testing sparse * dense:")
    try:
        result3 = matrix_multiply(A_sparse, B_dense)
        print(result3)
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Mixed case 2 - will fail
    print("Testing dense * sparse:")
    try:
        result4 = matrix_multiply(A_dense, B_sparse)
        print(result4)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_matrix_multiply()