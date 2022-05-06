import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader


def calc_num_correct(y_pred, y_true):
    return (y_pred.argmax(1) == y_true).type(torch.float).sum().item()


def train(
    model: nn.Module,
    train: DataLoader,
    loss_fn,
    optimizer: optim.Optimizer,
    device: str,
):
    size = len(train.dataset)
    model.train()
    acc = 0
    gamma = 0.75
    for batch, (X, y) in enumerate(train):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), batch * len(X)
        acc = (1 - gamma) * acc + gamma * calc_num_correct(pred, y) / len(
            y
        )  # running avg
        print(
            f"\rLoss: {loss:>7f} Accuracy: {100*acc:>0.1f}% [{current:>5d}/{size:>5d}]",
            end="",
        )
    print()


def test(model: nn.Module, test: DataLoader, loss_fn, device: str):
    size = len(test.dataset)
    num_batches = len(test)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
