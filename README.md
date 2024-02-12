# Fair k-center clustering

This repository contains several implementations of algorithms for fair k-center clustering.
For the latest version visit [the github repository](https://github.com/Cecca/fair-clustering).
The algorithms are implemented mostly in Python, with performance-sensitive parts implemented in Rust.

## Building the software

First, you need to [install Rust](https://rustup.rs/). 
Then, you can use [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html), 
you can setup the environment using the provided `env.yaml` file:

```
micromamba create -f env.yaml
micromamba activate fair-clustering
```

After that, you can compile the Rust extension used by the software using the following command:

```
maturing develop --release
```

## Datasets

All datasets used in the experimental evaluation are downloaded and preprocessed
when needed. See [`datasets.py`](https://github.com/Cecca/fair-clustering/blob/main/datasets.py)
for details.

If you want to download and preprocess all datasets in one go, you can use:

```
python datasets.py
```

Datasets are stored in [HDF5 files](https://www.hdfgroup.org/solutions/hdf5/) with the following structure:

- `data` contains the points of the dataset
- `data-PCA` contains the PCA rotation of the dataset (used mainly for visualization)
- `colors` is the vector of each point's colors

These data files can be processed using the very convenient [`h5py`](https://www.h5py.org/) library in Python.

## Running the experiments

The experiments reported in the paper are run using the code in the script [`experiments.py`](https://github.com/Cecca/fair-clustering/blob/main/experiments.py).
You need a copy of Cplex to run the experiments, so that then you can run

```
python experiments.py ~/your/path/to/cplex $EXPERIMENT
```

Where `$EXPERIMENT` is one of

- `exhaustive`
- `mapreduce`
- `streaming`

