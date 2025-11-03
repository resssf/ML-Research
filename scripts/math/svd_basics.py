import torch

# Matrix example
A = torch.randn(10, 8)

# SVD decomposition on PyTorch
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
S_matrix = torch.diag(S) # Covert from vector to diagonal matrix
A_reconstructed = U @ S_matrix @ Vh

print("U:\n", U)
print("S:\n", S)
print("Vh:\n", Vh)
print("A:\n", A)
print("A ~ U @ S @ Vh:\n", A_reconstructed)
