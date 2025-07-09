import torch
import random
from collections import defaultdict, Counter
import os
from NeuroGraph.datasets import NeuroGraphDataset, NeuroGraphDynamic

def download_neurograph(name: str, root: str = "data"):
    """
    Download and load a NeuroGraph dataset
    """
    os.makedirs(root, exist_ok=True)

    if name.startswith("Dyn"):
        dyn = NeuroGraphDynamic(root=root, name=name)
        dyn.download()          
        data = dyn.load_data()  
        return data              
    else:
        ds = NeuroGraphDataset(root=root, name=name)
        ds.download()            
        ds.process()             
        return ds     

def split_HCP_dataset(name: str, root: str = "data", train_ratio: float = 0.8, seed: int | None = None):
    """Download (if needed) an HCP-style dataset and split it
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs(root, exist_ok=True)
    cache_path = os.path.join(root, f"{name}.pt")

    if os.path.exists(cache_path):
        data = torch.load(cache_path)
    else:
        data = download_neurograph(name=name, root=root)

        if isinstance(data, dict) and {"labels", "batches"}.issubset(data.keys()):
            torch.save(data, cache_path)
        else:
            if hasattr(data, "labels") and hasattr(data, "batches"):
                torch.save({"labels": data.labels, "batches": data.batches}, cache_path)

    labels = data["labels"]
    batches = data["batches"]

    label_counts = Counter(labels)
    print("Label counts:", label_counts)

    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        indices_by_class[lbl].append(idx)

    train_indices: list[int] = []
    test_indices: list[int] = []

    for cls, indices in indices_by_class.items():
        random.shuffle(indices)
        split_point = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    train_dataset = [batches[i] for i in train_indices]
    test_dataset = [batches[i] for i in test_indices]

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    return train_dataset, test_dataset