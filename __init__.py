"""
Lung Disease Classification Package

This package contains modules for:
- Loading and preprocessing lung X-ray datasets
- Building and training CNN models (with and without hyperparameter tuning)
- Evaluating model performance
- Serving models locally and via Databricks
"""

from .config import *
from .data_preprocessing import load_dataset, split_dataset, preprocess_dataset
from .model import build_cnn_model
from .train_basic import train_basic_model
from .train_hyperopt import run_hyperopt
from .evaluation import evaluate_model, plot_history
from .serving import predict_local, predict_databricks
from .utils import visualize_samples

__all__ = [
    "load_dataset",
    "split_dataset",
    "preprocess_dataset",
    "build_cnn_model",
    "train_basic_model",
    "run_hyperopt",
    "evaluate_model",
    "plot_history",
    "predict_local",
    "predict_databricks",
    "visualize_samples"
]
