import argparse
from json import load
import pathlib
import os
import time

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from lowrank.pruners import PruningScope
from lowrank.pruners.alignment_pruner_gradient_based import AlignmentPrunerGradientBased
from lowrank.pruners.alignment_pruner_loss_based import AlignmentPrunerLossBased

import data_loader
import models
import trainer

PRUNERS = {"alignment_loss": AlignmentPrunerLossBased, "alignment_gradient": AlignmentPrunerGradientBased}

def main(
    dataset: str,
    model_name: str,
    pruner_type: str,
    sparsity: float,
    preprune_epochs: int,
    postprune_epochs: int,
    lr_step_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    device,
    load_saved_model: bool,
    prune_iterations: int = 1,
):
    device = torch.device(device)

    # create dataset, model, loss function, and optimizer
    train, test, num_classes = data_loader.get_data(dataset, batch_size=batch_size)
    model = models.all_models[model_name](batch_norm=True, num_classes=num_classes)
    model = model.to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    lr_schedule = lr_scheduler.StepLR(opt, step_size=lr_step_size, gamma=0.5)

    # check if model is saved (and load from save if necessary)
    checkpoint_dir = pathlib.Path("./checkpoints")
    if not checkpoint_dir.exists():
        os.makedirs(checkpoint_dir)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_model = checkpoint_dir / f"{model_name}.pt"
    if checkpoint_model.exists() and load_saved_model:
        print("Model found. Loading from checkpoint.")
        model.load_state_dict(torch.load(checkpoint_model))
        # sanity check by testing model performance
        trainer.test(model, test, loss_fn, device=device)
    else:
        checkpoint_model = checkpoint_dir / f"{model_name}_{timestr}.pt"
        print("Training from scratch. Model not found or --load_saved_model not passed.")
        for epoch in range(preprune_epochs):
            print(f"Pre-prune epoch {epoch+1} / {preprune_epochs}")
            trainer.train(model, train, loss_fn, opt, device=device)
            trainer.test(model, test, loss_fn, device=device)
            lr_schedule.step()
        print("Saving model")
        torch.save(model.state_dict(), checkpoint_model)

    # prune
    pruner = PRUNERS[pruner_type](
        device=device,
        model=models.convert_model_to_lr(model),
        scope=PruningScope.GLOBAL,
        sparsity=sparsity,
        dataloader=train,
        loss=loss_fn,
        prune_iterations=prune_iterations,
    )
    pruner.prune()
    model = model.to(device=device)

    # reduce LR by 2 post prune
    for g in opt.param_groups:
        g["lr"] /= 2

    # fine tune
    for epoch in range(postprune_epochs):
        print(f"Post-prune epoch {epoch+1} / {postprune_epochs}")
        trainer.train(model, train, loss_fn, opt, device=device)
        trainer.test(model, test, loss_fn, device=device)
        lr_schedule.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs a training session where a model is trained for some epochs, pruned, "
        "then trained for some more epochs"
    )
    parser.add_argument(
        "--model", type=str, choices=list(models.all_models.keys()), required=True
    )
    parser.add_argument(
        "--dataset", type=str, choices=list(data_loader.loaders.keys()), required=True
    )
    parser.add_argument(
        "--pruner", type=str, choices=list(PRUNERS.keys()), required=True
    )
    parser.add_argument("--sparsity", type=float)
    parser.add_argument("--preprune_epochs", type=int)
    parser.add_argument("--postprune_epochs", type=int)
    parser.add_argument("--prune_iterations", type=int)
    parser.add_argument("--lr_step_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--load_saved_model", action='store_true')
    parser.add_argument(
        "--device",
        choices=["cpu"] + ["cuda:" + str(i) for i in range(torch.cuda.device_count())],
        default="cpu",
    )
    args = parser.parse_args()
    main(
        model_name=args.model,
        dataset=args.dataset,
        pruner_type=args.pruner,
        sparsity=args.sparsity,
        preprune_epochs=args.preprune_epochs,
        postprune_epochs=args.postprune_epochs,
        lr_step_size=args.lr_step_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
        load_saved_model=args.load_saved_model,
        prune_iterations=args.prune_iterations
    )
