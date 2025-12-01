import torch
from src.model import MNISTClassifier
from src.dataloader import get_mnist_loaders
from src.evaluate import evaluate

def load_and_evaluate(model_path='checkpoints/mnist_model.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MNISTClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    _, test_loader = get_mnist_loaders()
    
    loss_function = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, loss_function, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    if test_acc >= 95.0:
        print("GOAL ACHIEVED: >95% accuracy!")
    else:
        print("Goal not yet reached.")
    
    return test_acc

if __name__ == "__main__":
    load_and_evaluate()
