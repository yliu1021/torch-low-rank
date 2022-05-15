import argparse
from json import load
import pathlib
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from lowrank.pruners import PruningScope
from lowrank.pruners.alignment_pruner_gradient_based import AlignmentPrunerGradientBased
from lowrank.pruners.alignment_pruner_loss_based import AlignmentPrunerLossBased
from lowrank.pruners.hybrid_pruner import HybridPruner

import data_loader
import models
import trainer

PRUNERS = {"alignment_loss": AlignmentPrunerLossBased, "alignment_gradient": AlignmentPrunerGradientBased, "hybrid": HybridPruner}
PRUNING_SCOPES = {"global": PruningScope.GLOBAL, "local": PruningScope.LOCAL}
MAX_EPOCHS = 160 

def main(
    dataset: str,
    model_name: str,
    pruner_type: str,
    sparsity: float,
    pre_prune_epochs: int,
    post_prune_epochs: int,
    lr_step_size: int,
    lr: float,
    scale_down_pruned_lr: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    device,
    pruning_scope: str,
    load_saved_model: bool,
    prune_iterations: int = 1,
    sparsity_bonus=1,
    data_path = "data",
    checkpoints_path = "checkpoints"
):
    device = torch.device(device)
    tb_writer = SummaryWriter()

    # create dataset, model, loss function, and optimizer
    train, test, num_classes = data_loader.get_data(dataset, batch_size=batch_size, data_path=data_path)
    model = models.all_models[model_name](batch_norm=True, num_classes=num_classes)
    model = model.to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    lr_schedule = lr_scheduler.StepLR(opt, step_size=lr_step_size, gamma=0.5)

    # check if model is saved (and load from save if necessary)
    checkpoint_dir = pathlib.Path(checkpoints_path)
    if not checkpoint_dir.exists():
        os.makedirs(checkpoint_dir)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    if pre_prune_epochs >= MAX_EPOCHS:
        checkpoint_model = checkpoint_dir / f"{model_name}.pt"
    else:
        checkpoint_model = checkpoint_dir / f"{model_name}_{str(pre_prune_epochs)}.pt"

    if checkpoint_model.exists() and load_saved_model:
        print(f"Model found at {checkpoint_model}. Loading from checkpoint.")
        model.load_state_dict(torch.load(checkpoint_model))
    else:
        print(f"Training from scratch. Model not found at {checkpoint_model} or --load_saved_model not passed.")
        for epoch in range(pre_prune_epochs):
            print(f"Pre-prune epoch {epoch+1} / {pre_prune_epochs}")
            trainer.train(model, train, loss_fn, opt, tb_writer=tb_writer, device=device, epoch=epoch)
            trainer.test(model, test, loss_fn, tb_writer=tb_writer, device=device, epoch=epoch)
            lr_schedule.step()
        print("Saving model")
        torch.save(model.state_dict(), checkpoint_model)

    # pre prune evaluate and log
    pre_prune_acc, pre_prune_loss = trainer.test(model, test, loss_fn, tb_writer=tb_writer, device=device)
    tb_writer.add_scalar("pre_prune_acc", pre_prune_acc)
    tb_writer.add_scalar("pre_prune_loss", pre_prune_loss)

    # prune
    for g in opt.param_groups:
        g["lr"] /= scale_down_pruned_lr
    model = models.convert_model_to_lr(model)
    pruners = {}
    for supported_pruner_type in PRUNERS.keys():
        if supported_pruner_type != "hybrid":
            pruners[supported_pruner_type] = PRUNERS[supported_pruner_type](
                device=device,
                model=model,
                scope=PRUNING_SCOPES[pruning_scope],
                sparsity=sparsity,
                dataloader=train,
                loss=loss_fn,
                opt=opt,
                prune_iterations=prune_iterations,
                sparsity_bonus=sparsity_bonus,
            )
    pruner = None
    if pruner_type == "hybrid":
        pruner = HybridPruner(
            pruners=list(pruners.values()),
            device=device,
            model=model,
            scope=PRUNING_SCOPES[pruning_scope],
            sparsity=sparsity,
            dataloader=train,
            loss=loss_fn,
            opt=opt,
            prune_iterations=prune_iterations,
            sparsity_bonus=sparsity_bonus,
        )
    else:
        pruner = pruners[pruner_type]
    pruner.prune()
    for i, layer in enumerate(pruner.layers_to_prune):
        tb_writer.add_scalar("effective_sparsity_per_layer", 1 - (layer.num_effective_params / layer.kernel_w.numel()), i)
    model = model.to(device=device)
    tb_writer.add_scalar("effective_sparsity", pruner.effective_sparsity())

    # post prune evaluate and log
    post_prune_acc, post_prune_loss =  trainer.test(model, test, loss_fn, tb_writer=tb_writer, device=device)
    tb_writer.add_scalar("post_prune_acc", post_prune_acc)
    tb_writer.add_scalar("post_prune_loss", post_prune_loss)

    # fine tune
    for epoch in range(post_prune_epochs):
        epoch += pre_prune_epochs
        print(f"Post-prune epoch {epoch+1} / {post_prune_epochs+pre_prune_epochs}")
        trainer.train(model, train, loss_fn, opt, tb_writer=tb_writer, device=device, epoch=epoch)
        trainer.test(model, test, loss_fn, tb_writer=tb_writer, device=device, epoch=epoch)
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
    parser.add_argument("--pre_prune_epochs", type=int)
    parser.add_argument("--post_prune_epochs", type=int)
    parser.add_argument("--prune_iterations", type=int)
    parser.add_argument("--lr_step_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--scale_down_pruned_lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--sparsity_bonus", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--pruning_scope", choices=list(PRUNING_SCOPES.keys()))
    parser.add_argument("--load_saved_model", action='store_true')
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--checkpoints_path", type=str)
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
        pre_prune_epochs=args.pre_prune_epochs,
        post_prune_epochs=args.post_prune_epochs,
        lr_step_size=args.lr_step_size,
        lr=args.lr,
        scale_down_pruned_lr=args.scale_down_pruned_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
        pruning_scope=args.pruning_scope,
        load_saved_model=args.load_saved_model,
        prune_iterations=args.prune_iterations,
        sparsity_bonus=args.sparsity_bonus,
        data_path=args.data_path,
        checkpoints_path=args.checkpoints_path
    )
