from calendar import c

import data_loader
from lowrank.pruners import PruningScope
import models
import torch
import trainer
from torch import nn, optim
from torch.optim import lr_scheduler

from lowrank.pruners.alignment_pruner_loss_based import AlignmentPrunerLossBased

def main(preprune_epochs: int, postprune_epochs: int, lr_drops: list[int], device):
    device = torch.device(device)
    train_dataloader, test_dataloader = data_loader.get_data("cifar10", batch_size=32)
    model = models.pytorch.vgg11(num_classes=10)
    model.to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_schedule = lr_scheduler.MultiStepLR(opt, milestones=lr_drops, gamma=0.1)

    for epoch in range(preprune_epochs):
        print(f"Pre-prune epoch {epoch}")
        trainer.train(model, train_dataloader, loss_fn, opt, device=device)
        trainer.test(model, test_dataloader, loss_fn, device=device)
        lr_schedule.step()

    # Prune
    pruner = AlignmentPrunerLossBased(
        model=models.convert_model_to_lr(model),
        scope=PruningScope.GLOBAL,
        sparsity=0.25,
        dataloader=train_dataloader,
        loss=loss_fn,
        prune_iterations=1
    )
    pruner.prune()

    # reduce LR by 2 post prune
    for g in opt.param_groups:
        g["lr"] /= 2

    for epoch in range(postprune_epochs):
        print(f"Pre-prune epoch {epoch}")
        trainer.train(model, train_dataloader, loss_fn, opt, device=device)
        trainer.test(model, test_dataloader, loss_fn, device=device)
        lr_schedule.step()


if __name__ == "__main__":
    main(preprune_epochs=0, postprune_epochs=1, lr_drops=[10], device='cuda')
