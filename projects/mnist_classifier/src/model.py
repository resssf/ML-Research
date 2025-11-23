import torch.nn as nn

class MNISTClassifier(nn.Module):
    """
    Convolutional Neural Network for image classification on the MNIST dataset.

    Architecture:
        - 2 convolutional layers (feature extraction)
        - 2 fully connected layers (classification)
        - ReLU activation after each convolution and fully connected layer
        - Output: logits for 10 classes

    Args:
        None

    Example usage:
        >>> from model import MNISTClassifier
        >>> model = MNISTClassifier()
        >>> x = torch.randn(32, 1, 28, 28)
        >>> logits = model(x)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3) # Convolution
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 24 * 24, out_features=128) # Fully connected
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Batch of input images (shape [batch_size, 1, 28, 28])

        Returns:
            torch.Tensor: Logits for classification (shape [batch_size, 10])
        """
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
