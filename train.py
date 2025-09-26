import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pprint

from src.config import Config
from src.logging_utils import setup_logging
from src.utils import set_seed, REGISTRY
from src.losses import _BUILTIN as _  # 触发注册
import src.metrics # 导入即可触发注册
from src.models.registry import build_model
from src.data.preprocess import preprocess
from src.data.dataset import build_dataloader
from src.data.create_dataloader import DataloaderSC
from src.data.transforms import standardize
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.resume import resume_from
from src.utils import best_checkpoint_path


def make_optimizer(params, cfg):
    name = cfg.optim.name
    if not hasattr(optim, name):
        raise ValueError(f"Unknown optimizer {name}")
    opt_cls = getattr(optim, name)
    return opt_cls(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay, betas=getattr(cfg.optim, 'betas', (0.9, 0.999)))


def make_scheduler(optimizer, cfg):
    if not cfg.scheduler.name:
        return None
    if not hasattr(optim.lr_scheduler, cfg.scheduler.name):
        raise ValueError(f"Unknown scheduler {cfg.scheduler.name}")
    sch_cls = getattr(optim.lr_scheduler, cfg.scheduler.name)
    return sch_cls(optimizer, **cfg.scheduler.kwargs)


def make_loss(cfg):
    if cfg.loss.name not in REGISTRY["loss"]:
        raise ValueError(f"Unknown loss {cfg.loss.name}")
    return REGISTRY["loss"][cfg.loss.name](**cfg.loss.kwargs)


def make_metrics(cfg):
    ms = {}
    for m in cfg.metrics:
        if m.name not in REGISTRY["metric"]:
            raise ValueError(f"Unknown metric {m.name}")
        ms[m.name] = REGISTRY["metric"][m.name]
    return ms


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def main(args):
    cfg = Config.from_yaml(args.config)
    setup_logging(cfg.logging.level, cfg.logging.to_file, cfg.logging.log_dir)
    set_seed(cfg.seed)

    # 打印所有参数配置
    logging.info("==== 配置参数 ====")
    logging.info("\n" + pprint.pformat(cfg, indent=2, width=100, compact=False))
    logging.info("==================")

    # 可选：预处理
    if args.preprocess:
        paths = preprocess(cfg.preprocess)
        logging.info("Preprocessed: %s", paths)

    # 数据
    dataloader, _ = DataloaderSC(data_cfg=cfg.data, model_path=cfg.model.model_path)
    
    dataset_name = cfg.data.dataset_name
    train_loader = dataloader[dataset_name]["train"]
    val_loader = dataloader[dataset_name]["dev"]
    test_loader = dataloader[dataset_name]["test"]
    print(len(train_loader))
    # 模型
    model = build_model(cfg.model.name, cfg.model.import_path, cfg.model.class_name, **cfg.model.kwargs)
    device = resolve_device(cfg.train.device)
    model.to(device)

    # 优化器/调度器/损失/指标
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(trainable_params, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    loss_fn = make_loss(cfg)
    metrics = make_metrics(cfg)

    # 断点续训（如提供）
    if args.resume:
        resume_from(args.resume, model, optimizer, scheduler, device)

    # 训练
    trainer = Trainer(model, optimizer, scheduler, loss_fn, metrics, device, cfg)
    history = trainer.fit(train_loader, val_loader)

    # 使用最佳权重评测
    best_path = best_checkpoint_path(cfg.train.save_dir)
    if os.path.exists(best_path):
        from src.utils import load_checkpoint
        ckpt = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])  # 使用最佳
        evaluator = Evaluator(model, loss_fn, metrics, device)
        test_logs = evaluator.evaluate(test_loader)
        logging.info("TEST: %s", {k: round(v, 5) for k, v in test_logs.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--preprocess", action="store_true", help="运行数据预处理")
    parser.add_argument("--resume", type=str, default=None, help="从 ckpt 路径续训")
    args = parser.parse_args()
    main(args)