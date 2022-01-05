# Low Rank Tensorflow

Low rank structures have been shown to demonstrate good
generalization abilities. This Python package aims to
implement some common TensorFlow layers with low rank
representations.

## Experiments

To run the experiments, you must first install the `lowrank`
package located in `src`.
For example, in the project root directory, you can run
```shell
pip install -e .
```

Then, the experiments themselves are located in `tests`. The modules
`data.py` and `model.py` are for creating the training/testing data
as well as the model. Results are stored in subdirectories and are
analyzed via the various `ipynb` notebooks.
