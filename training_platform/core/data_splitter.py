"""
Intelligent Data Splitting

Supports multiple splitting strategies:
- Random split (train/val/test)
- Stratified split (balanced by category)
- Time-series split (temporal ordering)
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


def random_split(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Random split into train/val/test sets.

    Args:
        data: List of data items
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility

    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": [data[i] for i in indices[:train_end]],
        "val": [data[i] for i in indices[train_end:val_end]],
        "test": [data[i] for i in indices[val_end:]],
    }


def stratified_split(
    data: List[Dict],
    category_field: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Stratified split maintaining category balance in each split.

    Args:
        data: List of data items
        category_field: Field name to stratify by
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        {"train": [...], "val": [...], "test": [...], "category_distribution": {...}}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

    # Group by category
    groups: Dict[str, List[int]] = {}
    for i, item in enumerate(data):
        cat = str(item.get(category_field, "unknown"))
        groups.setdefault(cat, []).append(i)

    rng = random.Random(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    for cat, indices in groups.items():
        rng.shuffle(indices)
        n = len(indices)
        train_end = max(1, int(n * train_ratio))  # At least 1 in train
        val_end = train_end + max(0, int(n * val_ratio))

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])

    # Shuffle within each split
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    # Category distribution
    train_cats = Counter(str(data[i].get(category_field, "unknown")) for i in train_indices)
    val_cats = Counter(str(data[i].get(category_field, "unknown")) for i in val_indices)
    test_cats = Counter(str(data[i].get(category_field, "unknown")) for i in test_indices)

    return {
        "train": [data[i] for i in train_indices],
        "val": [data[i] for i in val_indices],
        "test": [data[i] for i in test_indices],
        "category_distribution": {
            "train": dict(train_cats),
            "val": dict(val_cats),
            "test": dict(test_cats),
        },
    }


def temporal_split(
    data: List[Dict],
    time_field: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, List[Dict]]:
    """
    Time-series split preserving temporal order.
    Earlier data goes to train, later to val/test.

    Args:
        data: List of data items
        time_field: Field name containing timestamp or ordering key
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

    # Sort by time field
    sorted_data = sorted(data, key=lambda x: str(x.get(time_field, "")))

    n = len(sorted_data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": sorted_data[:train_end],
        "val": sorted_data[train_end:val_end],
        "test": sorted_data[val_end:],
    }


def split_data(
    data: List[Dict],
    method: str = "random",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    category_field: str = None,
    time_field: str = None,
) -> Dict[str, Any]:
    """
    Split data using the specified method.

    Args:
        data: List of data items
        method: 'random', 'stratified', 'temporal'
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
        category_field: For stratified split
        time_field: For temporal split

    Returns:
        Split results with statistics
    """
    if method == "random":
        result = random_split(data, train_ratio, val_ratio, test_ratio, seed)
    elif method == "stratified":
        if not category_field:
            raise ValueError("category_field required for stratified split")
        result = stratified_split(data, category_field, train_ratio, val_ratio, test_ratio, seed)
    elif method == "temporal":
        if not time_field:
            raise ValueError("time_field required for temporal split")
        result = temporal_split(data, time_field, train_ratio, val_ratio, test_ratio)
    else:
        raise ValueError(f"Unknown split method: {method}")

    # Add statistics
    result["statistics"] = {
        "total": len(data),
        "train_count": len(result["train"]),
        "val_count": len(result["val"]),
        "test_count": len(result["test"]),
        "train_ratio": len(result["train"]) / len(data) if data else 0,
        "val_ratio": len(result["val"]) / len(data) if data else 0,
        "test_ratio": len(result["test"]) / len(data) if data else 0,
        "method": method,
    }

    return result


def split_and_save(
    data: List[Dict],
    output_dir: str,
    method: str = "random",
    output_format: str = "jsonl",
    **kwargs,
) -> Dict[str, Any]:
    """Split data and save to files."""
    from .data_converter import _save_data

    result = split_data(data, method, **kwargs)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {}
    for split_name in ["train", "val", "test"]:
        file_path = str(output_path / f"{split_name}.{output_format}")
        _save_data(result[split_name], file_path)
        files[split_name] = file_path

    return {
        "files": files,
        "statistics": result["statistics"],
    }
