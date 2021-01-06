# project-template

This repository is a template for methodological projects, which require evaluating
multiple methods across a range of tasks (defined by problem size, signal-to-noise ratio, etc.),
and possibly across a range of metrics (e.g., accuracy and computation time). This repository
contains usage examples for many helpful packages and useful practices. 

The packages include:
* `tqdm` for progress bars. See usage in `experiments/result_manager.py`.
* `line_profiler` for profiling code. See usage in `scratch/profile_mean_vs_std.py`.
* `seaborn` for prettier plotting. See usage in `experiments/evaluation_manager.py`.
* `itertools` for simplifying for-loops/list comprehensions.

Some useful practices include:
* A *ResultManager* which saves/loads the results of running different (potentially time-consuming) algorithms.
* An *EvaluationManager* which handles plotting results.
* A `setup.sh` file which helps make the code easier to reproduce.
* Don't use Jupyter notebooks too heavily - they lead to a pretty messy code base. An example
of how I might use it to generate some "final" figures is in `experiments/notebooks/example-notebook.ipynb`.
* Code/data/result/figure organization.

To see how these are put into use, run:
* `python3 experiments/result_manager.py`
* `python3 experiments/evaluation_manager.py`

The below section shows what I'd usually put into the README.

## Main README:

This repository contains the code for **project** (link to paper).

To install the necessary requirements in a virtual environment and activate it, run:
```
bash setup.sh
source venv/bin/activate
```

The repository is organized as follows:
* `/algorithms` contains implementations of each algorithm.
* `/experiments` contains the code to replicate experiments.
* `/scratch` contains files with "scratch work", e.g. profiling and other testing.