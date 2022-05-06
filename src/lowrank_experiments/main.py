from torch import optim
from torch import nn

import data_loader
import models
import trainer


def main():
    train, test = data_loader.get_data("cifar10", batch_size=128)
    model = models.vgg11(num_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for _ in range(5):
        trainer.train(model, train, loss_fn, opt, device="cpu")
        trainer.test(model, test, loss_fn, device="cpu")


if __name__ == "__main__":
    main()
