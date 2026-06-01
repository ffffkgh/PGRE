# PGRE: Poisson--Gamma Modeling for Dynamic Knowledge Graph Relation Evolution

This repository contains the experimental implementation of **PGRE**, a probabilistic model for dynamic knowledge graph link prediction. The model is designed to capture temporal evolution patterns and inter-relational dependencies in dynamic knowledge graphs through a Poisson--Gamma Bayesian modeling framework.

The code in this repository is mainly used for the experiments reported in our PGRE paper, including link prediction on temporal knowledge graph datasets and comparative experiments with related probabilistic dynamic count models.

## Overview

Dynamic knowledge graphs contain facts represented as temporal quadruples:

```text
head_entity  relation  tail_entity  timestamp
```

PGRE models the generation and evolution of relation-specific adjacency tensors over time. Given a sequence of temporal relational graphs, the model learns latent entity-community factors, relation-specific intensity variables, and relation transition structures, and then predicts future links.

The repository currently supports experiments on sliced temporal knowledge graph datasets, including:

- `ICEWS18`
- `WIKI`
- `GDELT`

## Repository Structure

```text
PGRE-main/
├── data/
│   ├── GDELT/
│   ├── ICEWS18/
│   └── WIKI/
├── models.py                         # Main PGRE-related probabilistic model implementation
├── run_t.py                          # Main script for PGRE link prediction experiments
├── utils.py                          # Data loading, tensor construction, and evaluation utilities
├── GS_DPGM.py                        # Dynamic Poisson-Gamma baseline components
├── PGDS_Interval_Tensor.py           # PGDS tensor model
├── PRGDS_Interval_Tensor.py          # PRGDS tensor model
├── GS_NBRGDS_Interval_Tensor.py      # NBRGDS tensor model
├── _pgds_info_rate.py                # PGDS experimental script
├── _prgds_info_rate.py               # PRGDS experimental script
├── _nbrds_info_rate.py               # NBRGDS experimental script
└── src/                              # Auxiliary APF/Cython-based components and tests
```

## Environment Setup

Create a new Conda environment:

```bash
conda create -n pgre python=3.9
conda activate pgre
```

Install the required Python packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm openpyxl psutil path.py joblib cython
pip install torch
```

If you maintain a `requirements.txt` file, the environment can also be installed with:

```bash
pip install -r requirements.txt
```

### Notes on Computation

PGRE is a Bayesian probabilistic model based on MCMC-style posterior inference. The main inference procedure is CPU-oriented. GPU acceleration is generally not directly applicable to the sampling procedure, although GPU-enabled PyTorch can still be installed for compatibility with tensor operations or neural-network-based comparative methods.

## Datasets

The datasets should be placed under the `data/` directory. Each dataset folder should contain the following files:

```text
data/DATASET_NAME/
├── train_sliced.txt
├── valid_sliced.txt
├── test_sliced.txt
├── entity2id.txt
├── relation2id.txt
├── entity_map.txt
├── relation_map.txt
└── stat_sliced.txt
```

For example:

```text
data/ICEWS18/train_sliced.txt
data/ICEWS18/valid_sliced.txt
data/ICEWS18/test_sliced.txt
```


## Running PGRE Experiments

The main PGRE experiment can be launched by running:

```bash
python run_t.py
```

By default, the script uses the following configuration inside `run_t.py`:

```python
data_name = "GDELT"
burnin_epochs = 40
collection_epochs = 20
R = 8
```

To run experiments on another dataset, modify the following lines in `run_t.py`:

```python
data_name = "ICEWS18"   # Options: "GDELT", "ICEWS18", "WIKI"
R = 10                   # Use 8 for GDELT, 10 for ICEWS18, 15 for WIKI by default
```

The script will:

1. Load `train_sliced.txt`, `valid_sliced.txt`, and `test_sliced.txt`.
2. Select the top-`R` relations according to training frequency.
3. Construct a 4D adjacency tensor with shape:

```text
R × T × N × N
```

4. Train the PGRE model on historical time slices.
5. Save the learned parameters.
6. Evaluate future link prediction performance on the test slice.

Model checkpoints are saved to:

```text
load_model/trained_model_{DATASET}_R_{R}_{burnin}_{collection}.pth
```

If the directory does not exist, create it before running the script:

```bash
mkdir -p load_model
```

## Running Comparative Probabilistic Models

This repository also includes scripts for related dynamic count models used in comparative experiments.

### PGDS

```bash
python _pgds_info_rate.py
```

### PRGDS

```bash
python _prgds_info_rate.py
```

### NBRGDS

```bash
python _nbrds_info_rate.py
```

The default dataset path and hyperparameters are specified in the `params` dictionary at the bottom of each script. For example:

```python
params = {
    'data_dir': 'data/icews_tensor_preprocessed_33610010013.npz',
    'K': 100,
    'S': 3,
    'burin': 80,
    'maxiter': 20,
    'type': 'tensor',
    'seed': None
}
```

Modify `data_dir`, `K`, `S`, `burin`, and `maxiter` according to the dataset and experimental setting.

## Evaluation Metrics

The link prediction experiments evaluate predicted links using standard binary prediction metrics, including:

- ROC-AUC
- PR-AUC
- F1 score
- Recall
- Best-F1
- Best precision
- Best recall
- Best threshold

For count-model-based experiments, the scripts also report forecasting and smoothing metrics such as:

- Kullback--Leibler error
- Ranked probability score / DRPS
- Mean absolute error
- Mean relative error
- Negative log-likelihood



## Acknowledgements

This project builds on probabilistic modeling ideas for dynamic count data and temporal relational modeling. We also thank the authors of related open-source implementations and baseline methods that support comparative evaluation in dynamic graph and dynamic knowledge graph research.
