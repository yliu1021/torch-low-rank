from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


loaders = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}


def get_data(name: str, batch_size: int = 128):
    if name not in loaders:
        raise ValueError(f"Invalid dataset {name}")
    data = loaders[name]
    train = data(root=f"data", train=True, transform=ToTensor(), download=True)
    test = data(root=f"data", train=False, transform=ToTensor(), download=True)
    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size)
    return train, test


if __name__ == "__main__":
    get_data("cifar100")
