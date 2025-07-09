from data.open import split_HCP_dataset
from models.neuro import DynamicGNN
from models.tempsmooth import DynamicSpatialTemporalClassifier

from utils.train import train_model

from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv,SAGEConv,GraphConv,TransformerConv,ChebConv,GATConv,SGConv,GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import yaml



with open("config.yaml", "r") as fp:
    cfg = yaml.safe_load(fp)



gender_dataset = cfg["datasets"]["gender"]
age_dataset    = cfg["datasets"]["age"]

train_gender, test_gender = split_HCP_dataset(gender_dataset)
train_age,    test_age    = split_HCP_dataset(age_dataset)


gender_train_loader = DataLoader(train_gender, batch_size=cfg["training"].get("batch_size", 1), shuffle=True)
gender_test_loader  = DataLoader(test_gender,  batch_size=cfg["training"].get("batch_size", 1), shuffle=False)

age_train_loader    = DataLoader(train_age,    batch_size=cfg["training"].get("batch_size", 1), shuffle=True)
age_test_loader     = DataLoader(test_age,     batch_size=cfg["training"].get("batch_size", 1), shuffle=False)



import torch_geometric.nn as tgnn


def _resolve_string(value: str):
    if isinstance(value, str):
        if hasattr(nn, value):
            return getattr(nn, value)
        if hasattr(tgnn, value):
            return getattr(tgnn, value)
    return value


def instantiate_model(model_cfg):
    model_type = model_cfg["type"]
    params = {k: _resolve_string(v) for k, v in model_cfg["params"].items()}

    if model_type == "DynamicSpatialTemporalClassifier":
        return DynamicSpatialTemporalClassifier(**params)
    elif model_type == "DynamicGNN":
        return DynamicGNN(**params)


models = {name: instantiate_model(m_cfg) for name, m_cfg in cfg["models"].items()}



training_kwargs = {
    "lr": cfg["training"].get("lr", 1e-3),
    "num_epochs": cfg["training"].get("num_epochs", 100),
    "tol": cfg["training"].get("tol", 1e-4),
}


def _select_loaders(model_name):
    lower = model_name.lower()
    if "gender" in lower:
        return gender_train_loader, gender_test_loader
    elif "age" in lower:
        return age_train_loader, age_test_loader


trained_models = {}
for name, model in models.items():
    train_loader, test_loader = _select_loaders(name)
    print(f"\n--- Training {name} ---")
    train_model(model, train_loader, test_loader, **training_kwargs)
    trained_models[name] = model

