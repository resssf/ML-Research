import time
import torch

def basic_multiplication(matrix, n):
    result = matrix.clone()
    for _ in range(n - 1):
        result = result @ matrix
    return result

def svd_multiplication(matrix, n):
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=True)
    S_pow = torch.diag(S ** n)
    return U @ S_pow @ Vh

def make_symmetric(size):
    matrix = torch.randn(size, size)
    return 0.5 * (matrix + matrix.T)

if __name__ == "__main__":
    print("—" * 80)
    print(f"{"Explanations and practice":^80}")
    print("—" * 80)

    print(
        "A = U S Vh\n"
        "U and Vh are orthonormal matrices (U^TU = I, Vh Vh^T = I).\n"
        "For symmetric A: Vh = U.T, but in general U and Vh are independent.\n"
        "S is a diagonal matrix, so S^TS = diag(S1^2, S2^2, ...)\n"
        "\n"
        "Let's take A — a square and symmetric matrix — and compute A^n.\n"
        "A^n = (U S Vh)^n = U (S^n) Vh (when A is symmetric: Vh = U.T)\n"
    )

    print("—" * 80)
    print(f"{"Comparing direct multiplication vs SVD powering":^80}")
    print("—" * 80)

    for size in range(30, 300, 10):
        n = 100000
        A = make_symmetric(size)

        # Direct multiplication
        start_time_basic = time.time()
        basic_result = basic_multiplication(A, n)
        end_time_basic = time.time()
        basic_time = end_time_basic - start_time_basic

        # SVD multiplication
        start_time_svd = time.time()
        svd_result = svd_multiplication(A, n)
        end_time_svd = time.time()
        svd_time = end_time_svd - start_time_svd

        print(f"\nA.shape: [{size}x{size}]")
        print(f"  Time spent on multiplication:")
        print(f"    Direct method:       {basic_time:.4f} seconds")
        print(f"    SVD decomposition:   {svd_time:.4f} seconds")
        print(f"  Ratio (Direct/SVD): {basic_time/svd_time:4f}")
    
    print("—" * 80)
    print("""Conclusion:
  If the pow values are small or the matrix is small, there is either no difference or 
the direct method is faster. However, once the pow value exceeds 1000, difference becomes not just
noticeable, but differing by many orders of magnitude. 
  This is one of the most important SVD properties.
""")