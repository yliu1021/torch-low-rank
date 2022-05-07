import argparse

import data_loader
import models
import torch
import trainer
from torch import nn, optim
from torch.optim import lr_scheduler


def main(
    preprune_epochs: int,
    postprune_epochs: int,
    lr_drops: list[int],
    lr: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    device,
):
    device = torch.device(device)
    train, test = data_loader.get_data("cifar10", batch_size=batch_size)
    model = models.pytorch.vgg11(num_classes=10)
    models.convert_module(model)
    model.to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
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
    parser.add_argument("--preprune_epochs", type=int)
    parser.add_argument("--postprune_epochs", type=int)
    parser.add_argument("--lr_drop", type=int, action="append")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(
        preprune_epochs=args.preprune_epochs,
        postprune_epochs=args.postprune_epochs,
        lr_drops=sorted(args.lr_drop),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
    )
