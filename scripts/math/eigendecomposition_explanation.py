import torch

# Generate a symmetric matrix (e.g., random)
A = torch.randn(3, 3)
A = (A + A.T) / 2  # symmetrize

# Compute eigenvalues and eigenvectors (eigendecomposition)
eigenvalues, eigenvectors = torch.linalg.eigh(A)

# Reconstruct the original matrix using eigendecomposition
# A_reconst = Q Λ Q^T
Q = eigenvectors
Lambda = torch.diag(eigenvalues)
A_reconst = Q @ Lambda @ Q.T

print("—" * 80)
print(f"{"Eigenvalues, Eigenvectors and Eigenvalue decomposition explanation":^80}")
print("—" * 80)
print("""
Eigenvalues and Eigenvectors:
    A v = λ v
where λ is an eigenvalue and v is an eigenvector.

Eigenvalue decomposition (eigendecomposition) expresses a square matrix A as:
    A = V Λ V^(-1)
where V contains the eigenvectors and Λ is a diagonal matrix of eigenvalues.

- For general matrices, eigenvalues and eigenvectors may be complex.
- The matrix V is invertible if A has a full set of linearly independent eigenvectors (i.e., is diagonalizable).
- Not all matrices are diagonalizable; some may not have enough eigenvectors to form V.

Geometric intuition:
- Each eigenvector points in a direction that stays unchanged (except for scaling) under the transformation by A.
- The eigenvalue tells how much the matrix stretches or compresses along the corresponding eigenvector.
- In eigendecomposition, transforming a vector by A is equivalent to scaling its components in the eigenvector basis.""")
print("—" * 80)
print("\nOriginal matrix A:", A)
print("\nMatrix reconstructed by eigendecomposition:", A_reconst)
print("\nmax|A - A_reconst| =", (A - A_reconst).abs().max())
