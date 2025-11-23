from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_mnist_loaders(
    root='../data/raw', 
    batch_size=64,
    shuffle_train=True, 
    shuffle_test=False,
    transform=transforms.ToTensor()
):
    """
    Loads MNIST dataset and returns train/test DataLoader objects.

    Args:
        root (str, optional): Directory path to store/download MNIST data. Default is '../data/raw'.
        batch_size (int, optional): Number of samples per batch to load. Default is 64.
        shuffle_train (bool, optional): If True, shuffles training data each epoch. Default is True.
        shuffle_test (bool, optional): If True, shuffles test data each epoch. Default is False.
        transform (callable, optional): Transform to apply to dataset samples. Default is transforms.ToTensor().

    Returns:
        tuple: (train_loader, test_loader)
            train_loader (torch.utils.data.DataLoader): DataLoader for MNIST train dataset.
            test_loader  (torch.utils.data.DataLoader): DataLoader for MNIST test dataset.

    Example usage:
        >>> from dataloader import get_mnist_loaders
        >>> train_loader, test_loader = get_mnist_loaders(batch_size=32)
        >>> for images, labels in train_loader:
        ...     # training code
    """
    # Loading a dataset
    mnist_train = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    mnist_test = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    # DataLoader for train/test
    train_loader = DataLoader(
        mnist_train, 
        batch_size=batch_size, 
        shuffle=shuffle_train
    )
    test_loader = DataLoader(
        mnist_test, 
        batch_size=batch_size, 
        shuffle=shuffle_test
    )

    return train_loader, test_loader 

# Performance check
if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders()
    images, lables = next(iter(train_loader))
    print(f"Train batch images shape: {images.shape}, labels shape: {lables.shape}")

