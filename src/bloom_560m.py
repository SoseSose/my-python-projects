# %%
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from torch.optim import SGD, AdamW, Optimizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "bigscience/bloom-560m"


@dataclass
class Bloom560m_tokenizer_params:
    trust_remote_code: bool = False
    padding_side: str = "left"  # マスク言語モデルなら左に


def get_bloom560m_tokenizer(save_dir: str) -> AutoTokenizer:
    tokenizer_save_path = Path(save_dir) / MODEL_NAME

    if (tokenizer_save_path / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_save_path,
            **asdict(Bloom560m_tokenizer_params()),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            **asdict(Bloom560m_tokenizer_params()),
        )
        tokenizer.save_pretrained(tokenizer_save_path)

    return tokenizer


@dataclass
class Bloom560m_params:
    device_map: str = "cuda"
    trust_remote_code: bool = True
    torch_dtype: torch.dtype = torch.float16
    use_cache: bool = True
    # use_flash_attention_2:bool=True


class Bloom560m(LightningModule):
    def __init__(
        self,
        save_dir: str,
        lr: float,
    ):
        super().__init__()
        self.lr = lr
        self.tokenizer = get_bloom560m_tokenizer(save_dir)
        self.save_dir = save_dir

        model_save_path = Path(self.save_dir) / MODEL_NAME
        if (model_save_path / "config.json").exists():
            self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                model_save_path,
                **asdict(Bloom560m_params()),
            )

        else:
            self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                **asdict(Bloom560m_params()),
            )
            self.model.save_pretrained(model_save_path)
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model.forward(**batch)

    def training_step(self, batch, batch_idx):
        model_answer_logits = self.model.forward(**batch)
        train_loss = model_answer_logits.loss

        # あまりにもモデルが学習するとtrain_lossがnan, infになるかもしれない
        # 学習を安定させるため、その場合は0.0に変更
        if torch.isnan(train_loss) or torch.isinf(train_loss):
            train_loss = torch.tensor(0.0, device=train_loss.device)

        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            prog_bar=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        model_answer_logits = self.model.forward(**batch)
        self.log("val_loss", model_answer_logits.loss)

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        
        model_answer_logits = self.model.forward(**batch)
        self.log("test_loss", model_answer_logits.loss)
        return model_answer_logits.loss

        predictions = torch.argmax(model_answer_logits.logits, dim=-1)

        not_mask_positions = torch.where(batch["input_ids"] != self.tokenizer.mask_token_id, 1, 0)
        print(not_mask_positions)

        # maskされていない部分のみで正解率を計算するため、mask_positionsが1のときのみpredictionsとbatch["labels"]を比較する
        predicted_labels = predictions[not_mask_positions]
        true_labels = batch["labels"][not_mask_positions]

        correct_predictions = (predicted_labels == true_labels).sum().item()
        total_predictions = not_mask_positions.sum().item()
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )

        self.log("test_acc", accuracy, on_epoch=True, prog_bar=True)

        return {"test_loss": model_answer_logits.loss, "test_acc": accuracy}

    def predict_step(self, batch, batch_idx):
        label = batch["input_ids"][:, -2]
        question = batch["input_ids"][:, :-2]
        question_str = self.tokenizer.batch_decode(question)
        label_str = self.tokenizer.batch_decode(label)

        output = self.model.generate(question, max_length=100)
        gen_text = self.tokenizer.batch_decode(output)
        return {"question": question_str, "gen_text": gen_text, "label": label_str}

    def configure_optimizers(self) -> Optimizer:
        oprimizer = SGD(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=0.03,
        )

        schedulers = get_linear_schedule_with_warmup(
            optimizer=oprimizer, num_warmup_steps=100, num_training_steps=1000
        )
        return [oprimizer], [schedulers]
