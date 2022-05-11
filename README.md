# Low Rank Tensorflow

Low rank structures have been shown to demonstrate good
generalization abilities. This Python package aims to
implement some common TensorFlow layers with low rank
representations.

Documentation: https://yliu1021.github.io/tensorflow-low-rank/build/html/

## Updating Documentation

Update documentation: ```sphinx-apidoc -o docs/source/ src -f -M``` in root directory

## Running experiments
1. Create quild queues
```bash
for i in {1..10}; do guild run queue --background -y; done
```
2. Queue up staged runs
```bash
guild run set_rank prune_epoch=[0,10,20,50] pruner=[Magnitude,Alignment,WeightMagnitude,SNIP] pruning_scope=[local,global] sparsity=[0.75,0.9,0.95,0.98] total_epochs=128 lr=[0.01,0.05] l2=[0.0005,0.00005] model=vgg19 --stage-trials --tag="vgg19_2"
```

## Push / Pull Results

```bash
guild push gist:sjoshi804/low_rank_pruning_results.md
```

```bash
guild pull gist:sjoshi804/low_rank_pruning_results.md
```

## Testing

### Using Guild

VGG16 - CIFAR10

```bash 
guild run set_rank dataset=cifar10 pruner=Alignment_Loss prune_epoch=300 total_epochs=301 batch_size=256 sparsity=0.95 pruning_scope=global lr=0.05 model=vgg16 lr_scheduler_step_size=60 gpu=0
```

### Without Guild 
Running a vgg16 training run on CIFAR10 for 50 epochs (no pruning)
```bash
python ./src/lowrank_experiments/main.py --model=vgg16 --dataset=cifar10 --pre_prune_epochs=160 --post_prune_epochs=10 --lr_step_size=30 --lr=0.05 --momentum=0.9 --weight_decay=5e-4 --sparsity=0.9 --pruner=alignment_loss --pruning_scope=global --prune_iterations=20 --scale_down_pruned_lr=2 --load_saved_model --data_path=./data --checkpoints_path=./checkpoints --batch_size=256 --device=cuda:0
```
