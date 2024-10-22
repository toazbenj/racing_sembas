import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.nn import Module


from matplotlib.axes import Axes
from numpy import ndarray

from fut_data import FutData


def loss_graph(test_loss: ndarray, train_loss: ndarray, ax: Axes = None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Loss (KL Divergence)")

    ax.plot(np.arange(len(test_loss)), test_loss, color="red", label="Test Loss")
    ax.plot(np.arange(len(train_loss)), train_loss, color="blue", label="Train Loss")

    ax.set_title("Loss Graph")
    ax.legend()


def sample_graph(samples: list[tuple[ndarray, bool]], ax: Axes = None):
    if ax is None:
        fig, ax = plt.subplots()

    ts = np.array([p for p, cls in samples if cls])
    xs = np.array([p for p, cls in samples if not cls])

    if len(xs) > 0:
        ax.scatter(*xs.T, color="blue", marker=".", label="Non-Target Samples")
    if len(ts) > 0:
        ax.scatter(*ts.T, color="red", marker=".", label="Target Samples")

    ax.set_title("Sample Graph")
    ax.legend()


def brute_force_search(
    network: Module, dataset: FutData, classifier, n=100, ax: Axes = None
):
    if ax is None:
        fig, ax = plt.subplots()

    A, B = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    A = A.flatten()
    B = B.flatten()
    x = np.vstack((A, B)).T

    y_cls: torch.Tensor = classifier(
        network, dataset, torch.tensor(x, dtype=torch.float32)
    )

    ax.imshow(np.flip(y_cls.reshape((n, n)), 0))
    ax.set_title("Validity Graph")
