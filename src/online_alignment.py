import logging
import threading
import time
from collections import deque

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


class OnlineAlignmentManager:
    def __init__(self, model_name: str, lora_rank: int = 8, buffer_size: int = 500):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.model = get_peft_model(
            self.base_model,
            LoraConfig(r=lora_rank, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        )
        self.buffer = deque(maxlen=buffer_size)
        self.fisher_diag = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.lock = threading.Lock()
        self.is_training = False

    def collect(self, prompt: str, chosen: str, rejected: str, importance: float):
        with self.lock:
            self.buffer.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "weight": importance})

    def _compute_fisher(self, dataset: Dataset):
        self.model.train()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher_diag[n].zero_()
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            args=DPOConfig(output_dir="./tmp", max_steps=1, per_device_train_batch_size=2),
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.fisher_diag[n] += p.grad.detach() ** 2
        self.model.zero_grad()

    def update(self, ewc_lambda: float = 0.5, max_steps: int = 8):
        if len(self.buffer) < 16 or self.is_training:
            return
        self.is_training = True
        try:
            with self.lock:
                batch = sorted(self.buffer, key=lambda x: x["weight"], reverse=True)[:32]
                self.buffer.clear()
            dataset = Dataset.from_list(batch)
            self._compute_fisher(dataset)

            class EWC_DPOTrainer(DPOTrainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    loss = super().compute_loss(model, inputs, return_outputs)
                    ewc_loss = sum(
                        (self.fisher_diag[n] * (p - self.base_params[n]) ** 2).sum()
                        for n, p in model.named_parameters()
                        if p.requires_grad and n in self.fisher_diag
                    )
                    return loss + ewc_lambda * ewc_loss

            trainer = EWC_DPOTrainer(
                model=self.model,
                ref_model=None,
                args=DPOConfig(
                    output_dir="./lora_online",
                    max_steps=max_steps,
                    learning_rate=1e-4,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    logging_steps=2,
                ),
                train_dataset=dataset,
                tokenizer=self.tokenizer,
            )
            trainer.base_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
            trainer.fisher_diag = self.fisher_diag
            trainer.train()
            ckpt_path = f"./lora_ckpt/step_{int(time.time())}"
            self.model.save_pretrained(ckpt_path)
            logger.info(f"✅ Online DPO+EWC 完成 → {ckpt_path}")
        except Exception as e:
            logger.error(f"💥 对齐更新失败: {e}")
        finally:
            self.is_training = False

    def start_async_loop(self, interval: int = 300):
        def _loop():
            while True:
                time.sleep(interval)
                self.update()

        threading.Thread(target=_loop, daemon=True).start()
