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

Running a vgg11 training run on CIFAR10 for 50 epochs (no pruning)
```bash
python src/lowrank_experiments/experiments/set_rank.py --dataset=cifar10 --pruner=Alignment --prune_epoch=5 --total_epochs=50 --batch_size=128 --sparsity=0.25 --pruning_scope=local --lr=0.01 --l2=0.0005 --model=vgg11
```