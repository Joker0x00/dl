import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class BertPEFT(nn.Module):
    def __init__(self, num_classes: int, peft_r: int = 8, peft_alpha: int = 16, peft_dropout: float = 0.1, peft_target: list = ['query', 'value']):
        super().__init__()
        # 加载预训练 BERT
        self.bert = AutoModel.from_pretrained(r"C:\Users\wy\Desktop\models\bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,  # sequence‑level classification
                    inference_mode=False,
                    r=peft_r,
                    lora_alpha=peft_alpha,
                    lora_dropout=peft_dropout,
                    target_modules=peft_target,
                )
        self.bert = get_peft_model(self.bert, peft_config)
        # Freeze original BERT weights (trainable == LoRA + layernorm + pooler)
        for n, p in self.bert.named_parameters():
            if 'lora' not in n and 'classifier' not in n and 'LayerNorm' not in n:
                p.requires_grad = False
        self.bert.print_trainable_parameters()
        # 分类头
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] 向量
        return self.classifier(self.dropout(pooled))