import random
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from src.config import DataCfg

def _pick_col(df, preferred, aliases):
    """公共列名解析工具"""
    cols_lower = {c.lower(): c for c in df.columns}
    if preferred:
        k = preferred.lower()
        if k in cols_lower:
            return cols_lower[k]
    for a in aliases:
        k = a.lower()
        if k in cols_lower:
            return cols_lower[k]
    return None

def _standardize_examples(df, is_pair, column_map=None):
    """
    将任意列名标准化为 (text_a, text_b, label) 列表
    """
    column_map = column_map or {}

    text_a_aliases = ['text_a', 'sentence', 'sentence1', 'text', 'premise', 'query']
    text_b_aliases = ['text_b', 'sentence2', 'text2', 'hypothesis', 'response']
    label_aliases  = ['label', 'labels', 'gold', 'target', 'y']

    col_text_a = _pick_col(df, column_map.get('text_a'), text_a_aliases)
    col_label  = _pick_col(df, column_map.get('label'),  label_aliases)
    col_text_b = None
    if is_pair:
        col_text_b = _pick_col(df, column_map.get('text_b'), text_b_aliases)

    missing = []
    if col_text_a is None: missing.append("text_a")
    if col_label  is None: missing.append("label")
    if is_pair and col_text_b is None: missing.append("text_b")
    if missing:
        raise KeyError(
            f"Required columns not found: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df[col_text_a] = df[col_text_a].fillna("")
    if is_pair:
        df[col_text_b] = df[col_text_b].fillna("")

    examples = []
    if is_pair:
        for a, b, y in zip(df[col_text_a], df[col_text_b], df[col_label]):
            examples.append((str(a), str(b), y))
    else:
        for a, y in zip(df[col_text_a], df[col_label]):
            examples.append((str(a), None, y))
    return examples

# ---------- 各后缀的默认解析函数 ----------
def load_parquet(file_path, is_pair=False, column_map=None):
    import pandas as pd
    df = pd.read_parquet(file_path)
    return _standardize_examples(df, is_pair, column_map)

def load_tsv(file_path, is_pair=False, column_map=None):
    import pandas as pd
    df = pd.read_csv(file_path, sep="\t")
    return _standardize_examples(df, is_pair, column_map)

def load_csv(file_path, is_pair=False, column_map=None):
    import pandas as pd
    df = pd.read_csv(file_path)
    return _standardize_examples(df, is_pair, column_map)

# 默认注册表（可被外部覆盖/扩展）
DEFAULT_LOADER_REGISTRY = {
    ".parquet": load_parquet,
    "parquet": load_parquet,   # 兼容不带点
    ".tsv": load_tsv,
    "tsv": load_tsv,
    ".csv": load_csv,
    "csv": load_csv,
}

# ---------- 主入口 ----------
def DataloaderSC(data_cfg: DataCfg, model_path: str, loader_registry=None):
    """
    dataset_info 示例：
    {
        'sst2': {
            'data_dir': '/path/to/A1',
            'suffix': 'parquet',                # 或 '.tsv' / '.csv'
            'is_pair': False,
            'labels': ['0', '1'],
            'column_map': {'text_a':'sentence','label':'label'}  # 可选
        },
        'ymrpc': {
            'data_dir': '/path/to/B1',
            'suffix': 'parquet',
            'is_pair': True,
            'labels': ['0', '1'],
            'column_map': {'text_a':'sentence1','text_b':'sentence2','label':'label'}  # 可选
        }
    }
    返回: dataloader, iter_dataloader, label_list_map
    """
    registry = dict(DEFAULT_LOADER_REGISTRY)
    if loader_registry:
        registry.update(loader_registry)  # 外部可注入或覆盖解析器
    # todo 先这样吧
    tokenizer = BertTokenizer.from_pretrained(r"C:\Users\wy\Desktop\models\bert-base-uncased", do_lower_case=False, local_files_only=True)

    dataloader, label_list_map = {}, {}

    ext = data_cfg.suffix
    ext_key = ext if ext.startswith(".") else f".{ext}"
    # 找不到就尝试不带点
    loader = registry.get(ext_key) or registry.get(ext)
    if loader is None:
        raise ValueError(f"No loader registered for suffix '{ext}'. "
                            f"Known: {list(registry.keys())}")

    label_list = data_cfg.labels
    label2id = {str(lbl): i for i, lbl in enumerate(label_list)}
    label_list_map[data_cfg.dataset_name] = label_list

    dataloader[data_cfg.dataset_name] = {}

    for mode in data_cfg.model_list:
        file_path = f"{data_cfg.data_dir}/{mode}{ext_key}"
        examples = loader(
            file_path,
            is_pair=data_cfg.is_pair,
            column_map=data_cfg.column_map
        )

        if mode == "train" and data_cfg.small_train is not None:
            random.shuffle(examples)
            examples = examples[:data_cfg.small_train]
        input_ids, attn_masks, type_ids, y_ids = [], [], [], []
        for a, b, y in examples:
            enc = tokenizer.encode_plus(
                a, b,
                add_special_tokens=True,
                max_length=data_cfg.max_seq_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_token_type_ids=True
            )
            input_ids.append(enc["input_ids"])
            attn_masks.append(enc["attention_mask"])
            # 某些模型可能没有 token_type_ids，这里兜底成 0 向量
            type_ids.append(enc.get("token_type_ids", [0]*data_cfg.max_seq_length))

            s = str(y)
            if s not in label2id:
                try:
                    s = str(int(y))
                except Exception:
                    pass
            if s not in label2id:
                raise KeyError(f"Label '{y}' not in label set {label_list}")
            y_ids.append(label2id[s])

        dataset = TensorDataset(
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn_masks, dtype=torch.long),
            torch.tensor(type_ids, dtype=torch.long),
            torch.tensor(y_ids, dtype=torch.long),
        )
        sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)
        dataloader[data_cfg.dataset_name][mode] = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            drop_last=(mode == "train"),
        )

    return dataloader, label_list_map
