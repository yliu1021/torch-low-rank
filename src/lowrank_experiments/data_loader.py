from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def get_data(name: str, batch_size: int = 128):
    if name == "cifar10":
        data = datasets.CIFAR10
    elif name == "cifar100":
        data = datasets.CIFAR100
    else:
        raise ValueError(f"Invalid dataset {name}")
    train = data(root=f"data", train=True, transform=ToTensor(), download=True)
    test = data(root=f"data", train=False, transform=ToTensor(), download=True)
    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size)
    return train, test


if __name__ == "__main__":
    get_data("cifar100")
