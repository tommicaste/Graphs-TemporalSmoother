# Graph based Temporal Smoother Classification

This repository introduces **Graph based Temporal Smoother (GTS)**, a new model for subject level prediction on **dynamic** brain network time series. GTS explicitly smooths node embeddings across adjacent snapshots before classification. For comparison we also include **NeuroGraph Dynamic**, a baseline derived from the NeuroGraph benchmark adapted to the same task.

## Overview

The codebase lets you train, evaluate, and compare two alternatives:

* **Graph based Temporal Smoother**: our proposed architecture that alternates spatial graph convolutions with a lightweight temporal transformer to share information over time.  
* **NeuroGraph Dynamic**: a re implementation of the NeuroGraph model configured for dynamic data that serves as a strong baseline.  

The pipeline automatically downloads NeuroGraph dynamic datasets, performs subject aware splits, and stores every artefact needed for reproducible experiments. All hyper parameters live in a single `config.yaml` file.

## Project Structure
```text
.
├── config.yaml
├── data/
│   └── open.py
├── models/
│   ├── tempsmooth.py
│   └── neuro.py
├── utils/
│   └── train.py
├── main.py
├── TemporalSmoother_print.pdf
├── requirements.txt
└── .gitignore
```

## Installation
```bash
git clone https://github.com/tommicaste/Graphs-TemporalSmoother.git
cd Graphs-TemporalSmoother

pip install -r requirements.txt
pip install -e .
```

## Usage

1. **Prepare data**: set the dataset tag in `config.yaml`, for example `DynHCPGender`.  
2. **Adjust the YAML**: enable or disable models in `models.*` and tune training settings such as `training.lr` and `num_epochs`.  
3. **Launch training**
   ```bash
   python main.py
   ```
   The script trains every model declared in the YAML and writes outputs under `runs/<model_name>/`.

## Evaluation

After training the pipeline reloads the checkpoint with the best validation accuracy and

* computes test metrics,  
* prints a concise log to stdout,  
* saves `metrics.json` and confusion matrices to `runs/<model_name>/results/`.

## Citations

* **NeuroGraph** — Anwar Said, Roza G. Bayrak, Tyler Derr, Mudassir Shabbir, Daniel Moyer, Catie Chang, and Xenofon Koutsoukos. “NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics.” arXiv preprint arXiv:2306.06202, 2024.  
