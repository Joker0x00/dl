from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml

@dataclass
class LoggingCfg:
    level: str = "INFO"
    to_file: bool = False
    log_dir: str = "logs"

@dataclass
class PreprocessCfg:
    input_path: str = "data/raw"
    output_path: str = "data/processed"
    format: str = "csv"
    target_column: str = "label"
    text_column: str = "text"
    image_dir: Optional[str] = None

@dataclass
class DataCfg:
    dataset_name: str
    data_dir: str
    suffix: str
    is_pair: bool
    labels: list
    small_train: int
    column_map: Dict[str, str]
    max_seq_length: int
    model_list: list[str]
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True

@dataclass
class ModelCfg:
    name: str
    model_path: str
    import_path: Optional[str] = None
    class_name: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimCfg:
    name: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

@dataclass
class SchedulerCfg:
    name: Optional[str] = "CosineAnnealingLR"
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainCfg:
    epochs: int = 20
    device: str = "cuda"
    grad_clip_norm: Optional[float] = None
    mixed_precision: bool = True
    save_dir: str = "checkpoints"
    save_every_epoch: bool = True
    monitor: str = "val/accuracy"
    monitor_mode: str = "max"
    early_stop_patience: Optional[int] = 5

@dataclass
class LossCfg:
    name: str = "CrossEntropyLoss"
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricItem:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Config:
    project: str = "demo-classification"
    seed: int = 42
    task: str = "test"
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    metrics: List[MetricItem] = field(default_factory=list)

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # 简单方式：用解包构造子对象
        def sub(key, cls):
            return cls(**data.get(key, {}))
        cfg = Config(
            project=data.get("project", "demo"),
            seed=int(data.get("seed", 42)),
            logging=sub("logging", LoggingCfg),
            preprocess=sub("preprocess", PreprocessCfg),
            data=sub("data", DataCfg),
            model=sub("model", ModelCfg),
            optim=sub("optim", OptimCfg),
            scheduler=sub("scheduler", SchedulerCfg),
            train=sub("train", TrainCfg),
            loss=sub("loss", LossCfg),
            metrics=[MetricItem(**m) for m in data.get("metrics", [])],
        )
        return cfg