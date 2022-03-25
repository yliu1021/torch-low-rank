# Low Rank Tensorflow

Low rank structures have been shown to demonstrate good
generalization abilities. This Python package aims to
implement some common TensorFlow layers with low rank
representations.

Documentation: https://yliu1021.github.io/tensorflow-low-rank

Update documentation: ```sphinx-apidoc -o docs/source/ . -f -M``` in root directory

## Running experiments
1. Create quild queues
```bash
for i in {1..10}; do guild run queue --background -y; done
```
2. Queue up staged runs
```bash
guild run set_rank prune_epoch=[1,2,5,10] pruner=[Magnitude,SNIP,Alignment] pruning_scope=[local,global] sparsity=[0.25,0.5,0.75] total_epochs=50 --trials=216 --stage-trials
```

## Push / Pull Results

```bash
guild push gist:sjoshi804/low_rank_pruning_results.md
```

```bash
guild pull gist:sjoshi804/low_rank_pruning_results.md
```