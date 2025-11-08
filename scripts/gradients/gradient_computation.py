"""
Gradient Computation: Manual vs PyTorch Autograd

Demonstration of gradient computation in two ways:
    1. Manual differentiation — mathematical
    2. Automatic differentiation — PyTorch autograd

Comparison of results shows that autograd produces accurate results.
"""

import torch
import numpy as np


def manual_gradient_computation():
    """
    Manual calculation of the gradient for a simple function.

    Loss function: loss = MSE(y_pred, y_true) = (1/n) * sum((y_pred - y_true)^2)

    Where: y_pred is a scalar from a linear model: y_pred = dot(x, w) + b

    Gradients:
    d(loss)/d(w) = d(loss)/d(y_pred) * d(y_pred)/d(w) = [ (2/n) * sum(y_pred - y_true) ] * x
    d(loss)/d(b) = d(loss)/d(y_pred) * d(y_pred)/d(b) = (2/n) * sum(y_pred - y_true)
    """
    # Data
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_true = np.array([1.0, 2.0, 0.0], dtype=np.float32)

    # Parameters
    w = np.array([0.5, 0.3, -0.2], dtype=np.float32)
    b = 0.1

    # Forward pass
    y_pred = np.dot(x, w) + b

    # Loss
    loss = np.mean((y_pred - y_true) ** 2)

    # Backward pass — manual calculation
    n = len(y_true)

    dloss_dypred = (2 / n) * np.sum(y_pred - y_true)
    dloss_dw = dloss_dypred * x
    dloss_db = dloss_dypred

    # output of the solution progress to the console
    print("—" * 80)
    print(f"\n1. MANUAL CALCULATION OF GRADIENTS")
    print(f"\nData:")
    print(f"   x = {x}")
    print(f"   y_true = {y_true}")
    print(f"\nParameters:")
    print(f"   w = {w}")
    print(f"   b = {b}")
    print(f"\nForward pass:")
    print(f"   y_pred = np.dot(x, w) + b = {y_pred:.4f}")
    print(f"\nLoss function:")
    print(f"   loss = np.mean((y_pred - y_true) ** 2) = {loss:.4f}")
    print(f"\nBackward pass:")
    print(f"   d(loss)/d(y_pred) = (2/n) * sum(y_pred - y_true) = {dloss_dypred:.4f}")
    print(f"   d(loss)/d(w) = d(loss)/d(y_pred) * x = {dloss_dw}")
    print(f"   d(loss)/d(b) = d(loss)/d(y_pred) * 1 = {dloss_db:.4f}")

    return {
        "loss": loss,
        "y_pred": y_pred,
        "dloss_dw": dloss_dw,
        "dloss_db": dloss_db,
        "x": x,
        "w": w,
        "b": b,
    }


def pytorch_gradient_computation(
    w_data, b_data, x_data=[1.0, 2.0, 3.0], y_data=[1.0, 2.0, 0.0]
):
    """
    Automatic computation of the gradient for a simple function with using PyTorch autograd.
    """
    # Converting data in PyTorch tensors
    x = torch.tensor(x_data, dtype=torch.float32)
    y_true = torch.tensor(y_data, dtype=torch.float32)

    # Parameters with requires_grad=True
    w = torch.tensor(w_data, dtype=torch.float32, requires_grad=True)
    b = torch.tensor(b_data, dtype=torch.float32, requires_grad=True)

    # Forward pass
    y_pred = torch.dot(x, w) + b
    y_pred.retain_grad()

    # Loss computation
    loss = torch.mean((y_pred - y_true) ** 2)

    # Backward pass — automatic computation
    loss.backward()

    # output
    print("—" * 80)
    print(f"\n2. AUTOMATIC COMPUTATION OF GRADIENTS (PyTorch Autograd)")
    print(f"\nData (PyTorch Tensors):")
    print(f"   x = {x}")
    print(f"   y_true = {y_true}")
    print(f"\nParameters (with requires_grad=True):")
    print(f"   w = {w} (requires_grad={b.requires_grad})")
    print(f"   b = {b} (requires_grad={b.requires_grad})")
    print(f"\nForward pass:")
    print(f"   y_pred = x * w + b = {y_pred.item():.4f}")
    print(f"   y_pred.grad_fn = {y_pred.grad_fn}")
    print(f"\nLoss function:")
    print(f"   loss = mean((y_pred - y_true) ^ 2) = {loss.item():.4f}")
    print(f"   loss.grad_fn = {loss.grad_fn}")
    print(f"\nBackward pass (loss.backward()):")
    if y_pred.grad is not None:
        print(f"   d(loss)/d(y_pred): y_pred.grad = {y_pred.grad.item():.4f}")
    print(f"   d(loss)/d(w): w.grad = {w.grad}")
    print(f"   d(loss)/d(b): b.grad = {b.grad}")

    return {
        "loss": loss.item(),
        "y_pred": y_pred.item(),
        "dloss_dw": w.grad.numpy(),
        "dloss_db": b.grad.item(),
        "x": x,
        "w": w,
        "b": b,
        
    }


def compare_results(manual_result, pytorch_result):
    tolerance = 1e-6
    are_all_close = all(
        [
            abs(manual_result["y_pred"] - pytorch_result["y_pred"]) <= tolerance,
            abs(manual_result["loss"] - pytorch_result["loss"]) <= tolerance,
            np.allclose(
                manual_result["dloss_dw"], pytorch_result["dloss_dw"], atol=tolerance
            ),
            abs(manual_result["dloss_db"] - pytorch_result["dloss_db"]) <= tolerance
        ]
    )

    print("—" * 80)
    print("\n3. COMPARING RESULTS")

    print(f"\nPrediction (y_pred):")
    print(f"   Manual:   {manual_result['y_pred']:.4f}")
    print(f"   PyTorch:  {pytorch_result['y_pred']:.4f}")

    print(f"\nLoss:")
    print(f"   Manual:   {manual_result['loss']:.4f}")
    print(f"   PyTorch:  {pytorch_result['loss']:.4f}")

    print(f"\ndloss/dw):")
    print(f"   Manual:   {manual_result['dloss_dw']}")
    print(f"   PyTorch:  {pytorch_result['dloss_dw']}")

    print(f"\ndloss/db):")
    print(f"   Manual:   {manual_result['dloss_db']:.4f}")
    print(f"   PyTorch:  {pytorch_result['dloss_db']:.4f}")

    print(f"\n{'SUCCESS' if are_all_close else 'FAILURE'} (tolerance: {tolerance})")
    return are_all_close


def advanced_example():
    """
    Advanced example: computing gradients of multiple layers.
    Network: x -> w1 -> ReLU -> w2 -> y_pred
    """
    print("—" * 80)
    print("\n4. ADVANCED EXAMPLE: MULTI-LAYER NETWORK")

    print(f"\nArchitecture:")
    print(f"   Layer 1: 3 → 2 (with ReLU)")
    print(f"   Layer 2: 2 → 1 (linear)")

    print(f"\nData:")
    x = torch.randn(3, requires_grad=False)
    y_true = torch.randn(1, requires_grad=False)
    print(f"   x = {x}")
    print(f"   y_true = {y_true}")

    print(f"\nParameters")
    w1 = torch.randn(3, 2, requires_grad=True)  # first layer
    b1 = torch.randn(2, requires_grad=True)
    w2 = torch.randn(2, 1, requires_grad=True)  # second layer
    b2 = torch.randn(1, requires_grad=True)
    print(f"   w1 = {w1}")
    print(f"   b1 = {b1}")
    print(f"   w2 = {w2}")
    print(f"   b2 = {b2}")

    print(f"\nForward pass:")
    # First layer
    z1 = torch.matmul(x, w1) + b1
    print(f"   z1 = x * w1 + b1")
    print(f"     z1.shape = {z1.shape}")
    print(f"     z1.grad_fn = {z1.grad_fn}")

    a1 = torch.relu(z1)  # ReLU
    print(f"   a1 = ReLU(z1)")
    print(f"     a1.shape = {a1.shape}")
    print(f"     a1.grad_fn = {a1.grad_fn}")

    # Second layer
    z2 = torch.matmul(a1, w2) + b2
    print(f"   z2 = a1 * w2 + b2")
    print(f"     z2.shape = {z2.shape}")
    print(f"     z2.grad_fn = {z2.grad_fn}")

    print(f"\nLoss")
    loss = torch.mean((z2 - y_true) ** 2)
    print(f"   loss = mean((z2 - y_true) ^ 2) = {loss.item():.4f}")
    print(f"   loss.grad_fn = {loss.grad_fn}")

    print(f"\nBackward pass (loss.backward()):")
    loss.backward()

    print(f"   Gradients:")
    print(
        f"     d(loss)/(dw)1: w1.grad.shape = {w1.grad.shape}, ||w1.grad|| = {torch.norm(w1.grad):.6f}"
    )
    print(
        f"     d(loss)/(db)1: b1.grad.shape = {b1.grad.shape}, ||b1.grad|| = {torch.norm(b1.grad):.6f}"
    )
    print(
        f"     d(loss)/(dw)2: w2.grad.shape = {w2.grad.shape}, ||w2.grad|| = {torch.norm(w2.grad):.6f}"
    )
    print(
        f"     d(loss)/(db)2: b2.grad.shape = {b2.grad.shape}, ||b2.grad|| = {torch.norm(b2.grad):.6f}"
    )

    print(f"\n   Chain rule in action")
    print(f"     d(loss)/d(z2) -> d(loss)/d(a1) -> d(loss)/d(z1) -> d(loss)/d(w1)")


def backward_pass_details():
    """
    Description how backward pass works.
    """
    print("—" * 80)
    print(f"\n5. HOW BACKWARD PASS WORKS IN DETAILS")
    print(f"""
1. INITIALIZATION
   - Sets the gradient of the root node (loss) = 1.0
     
2. GRAPH TRAVERSAL IN REVERSE ORDER (topological sort)
   - Starting from loss, moves to leaf nodes
   - Calculates its local derivative for each node
     
3. APPLICATION OF THE CHAIN RULE
   - Combines local derivatives with already calculated derivatives
   - Formula: grad_input = grad_output * local_derivative

4. ACCUMULATION OF GRADIENTS
   - Adds calculated gradients to attributes .grad of leaf nodes
   - That 's why you need to call .zero_() before the new backward() 

Example for better understanding:
   Network y_pred = x * w + b
       
   Forward:
   x -> [*w] -> z -> [+b] -> y_pred -> [MSE] -> loss
       
       y_pred = x * w + b
       loss = (y_pred - y_true) ^ 2

   Backward: (CHAIN RULE :D)
   x <- [*w] <- z <- [+b] <- y_pred <- [MSE] <- loss

       grad_loss = d(loss)/d(loss) = 1.0
       grad_y_pred = d(loss)/d(y_pred) = grad_loss * d(loss)/d(y_pred) = 2 * (y_pred - y_true)
       grad_z = d(loss)/d(z) = d(loss)/d(y_pred) * d(y_pred)/d(z) = grad_y_pred * 1 = 2 * (y_pred - y_true)
       grad_b = d(loss)/d(b) = d(loss)/d(y_pred) * d(y_pred)/d(z) * d(z)/d(b) = grad_z * 1 = 2 * (y_pred - y_true)
       grad_w = d(loss)/d(w) = d(loss)/d(y_pred) * d(y_pred)/d(z) * d(z)/d(w) = grad_z * x = 2 * (y_pred - y_true) * x""")


if __name__ == "__main__":
    print("—" * 80)
    print(
        "Torch Autograd: Comparison of manual calculation and automatic computation".center(
            80
        )
    )

    # running all examples
    manual_result = manual_gradient_computation()

    pytorch_result = pytorch_gradient_computation(
        w_data=manual_result["w"], b_data=manual_result["b"]
    )

    compare_results(manual_result, pytorch_result)
    advanced_example()
    backward_pass_details()

    print("—" * 80)
    print("\n6. Practicle conclusion and notes")
    print("""
1. PyTorch autograd calculates gradients ACCURATELY
2. Autograd works for fuctions DIFFERENT complexity
3. Autograd applies the chain rule AUTOMATICALLY
4. Gradients are ONLY available on leaf nodes
5. Remember the ACCUMULATION of gradients       
""")
    print("\nThank you!\n")
