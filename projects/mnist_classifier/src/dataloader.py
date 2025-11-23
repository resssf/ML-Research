from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_mnist_loaders(
    root='../data/raw', 
    batch_size=64,
    shuffle_train=True, 
    shuffle_test=False,
    transform=transforms.ToTensor()
):
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
    print(f"Train batch images shape: {images.shape}, labels shape: {labels.shape}")

