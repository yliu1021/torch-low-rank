import argparse

import data_loader
import models
import torch
import trainer
from torch import nn, optim
from torch.optim import lr_scheduler


def main(preprune_epochs: int, postprune_epochs: int, lr_drops: list[int], device):
    device = torch.device(device)
    train, test = data_loader.get_data("cifar10", batch_size=128)
    model = models.pytorch.vgg11(num_classes=10)
    models.convert_module(model)
    model.to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_schedule = lr_scheduler.MultiStepLR(opt, milestones=lr_drops, gamma=0.1)

    for epoch in range(preprune_epochs):
        print(f"Pre-prune epoch {epoch}")
        trainer.train(model, train, loss_fn, opt, device=device)
        trainer.test(model, test, loss_fn, device=device)
        lr_schedule.step()

    # TODO: add pruning
    ...

    # reduce LR by 2 post prune
    for g in opt.param_groups:
        g["lr"] /= 2

    for epoch in range(postprune_epochs):
        print(f"Pre-prune epoch {epoch}")
        trainer.train(model, train, loss_fn, opt, device=device)
        trainer.test(model, test, loss_fn, device=device)
        lr_schedule.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs a training session where a model is trained for some epochs, pruned, "
                    "then trained for some more epochs"
    )
    parser.add_argument("preprune_epoch", type=int)
    parser.add_argument("postprune_epoch", type=int)
    parser.add_argument("lr_drops", type=)
    main()
