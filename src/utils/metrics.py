"""
Evaluation metrics for model performance assessment.

Provides common metrics for classification and detection tasks.
"""

import torch
import numpy as np
from typing import Union, Tuple


def calculate_accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: Predicted class indices or logits
        targets: Ground truth class indices

    Returns:
        Accuracy as float between 0 and 1

    Example:
        >>> preds = torch.tensor([0, 1, 2, 1])
        >>> targets = torch.tensor([0, 1, 1, 1])
        >>> calculate_accuracy(preds, targets)
        0.75
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # If predictions are logits, get argmax
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    correct = np.sum(predictions == targets)
    total = len(targets)

    return float(correct) / total if total > 0 else 0.0


def calculate_precision_recall(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate per-class precision and recall.

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        num_classes: Number of classes

    Returns:
        Tuple of (precision, recall) arrays of shape (num_classes,)

    Example:
        >>> preds = torch.tensor([0, 1, 2, 1])
        >>> targets = torch.tensor([0, 1, 1, 2])
        >>> precision, recall = calculate_precision_recall(preds, targets, 3)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)

    for cls in range(num_classes):
        # True positives
        tp = np.sum((predictions == cls) & (targets == cls))
        # False positives
        fp = np.sum((predictions == cls) & (targets != cls))
        # False negatives
        fn = np.sum((predictions != cls) & (targets == cls))

        # Calculate precision
        if tp + fp > 0:
            precision[cls] = tp / (tp + fp)

        # Calculate recall
        if tp + fn > 0:
            recall[cls] = tp / (tp + fn)

    return precision, recall


def calculate_f1_score(
    precision: Union[float, np.ndarray],
    recall: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate F1 score from precision and recall.

    Args:
        precision: Precision value(s)
        recall: Recall value(s)

    Returns:
        F1 score(s)

    Example:
        >>> f1 = calculate_f1_score(0.8, 0.9)
        >>> f1
        0.8470...
    """
    denominator = precision + recall

    if isinstance(denominator, np.ndarray):
        f1 = np.zeros_like(precision)
        mask = denominator > 0
        f1[mask] = 2 * (precision[mask] * recall[mask]) / denominator[mask]
        return f1
    else:
        if denominator > 0:
            return 2 * (precision * recall) / denominator
        return 0.0
