import os
import numpy as np
import pandas as pd
from typing import Tuple

from src.config import PreprocessCfg

# 这里给出一个“表格/CSV → npz” 的通用示例；
# 你可根据任务类型(NLP/CV/TS)自定义。

def split_train_val_test(df: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    n = len(df)
    idx = np.random.RandomState(seed).permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


def preprocess(cfg: PreprocessCfg):
    os.makedirs(cfg.output_path, exist_ok=True)

    if cfg.format == "csv":
        df = pd.read_csv(os.path.join(cfg.input_path, "data.csv"))
    elif cfg.format == "json":
        df = pd.read_json(os.path.join(cfg.input_path, "data.json"))
    else:
        raise ValueError("Unsupported format")

    # 假设存在文本列与标签列，示例：将文本用简单hash向量化（占位）。
    # 真实项目中可替换为 tokenizer / 特征抽取 / 图像读取等。
    def featurize(text: str, dim=300):
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        return rng.randn(dim).astype(np.float32)

    X = np.stack(df[cfg.text_column].astype(str).apply(lambda s: featurize(s)).values)
    y = df[cfg.target_column].astype(int).values

    train_df, val_df, test_df = split_train_val_test(df)
    X_train = X[train_df.index]
    y_train = y[train_df.index]
    X_val = X[val_df.index]
    y_val = y[val_df.index]
    X_test = X[test_df.index]
    y_test = y[test_df.index]

    np.savez_compressed(os.path.join(cfg.output_path, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(cfg.output_path, "val.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(cfg.output_path, "test.npz"), X=X_test, y=y_test)

    return {
        "train": os.path.join(cfg.output_path, "train.npz"),
        "val": os.path.join(cfg.output_path, "val.npz"),
        "test": os.path.join(cfg.output_path, "test.npz"),
    }