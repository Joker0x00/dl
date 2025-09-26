# src/data/transforms.py
import numpy as np

def standardize(x: np.ndarray):
    m = x.mean()
    s = x.std() + 1e-8
    return (x - m) / s