# 单任务学习

## 训练+评测
```sh
python train.py --config configs/default.yaml
```

## 断点续训

```sh
python train.py --config /root/code/dl/configs/default.yml --resume /root/autodl-tmp/single/checkpoints/best.pt
```