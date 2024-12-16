from my_utils.model.gpt import GPT
from my_utils.model.modules import IndependentAdamW, CosineAnnealingLR
import torch
from torch.optim import Optimizer
from lightning import LightningModule


class GPTLightningModel(LightningModule):
    def __init__(
        self,
        gpt: GPT,
        optimizer: Optimizer,
        lr_scheduler: CosineAnnealingLR,
    ):
        super().__init__()
        self.model: GPT = gpt
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    
    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor|None]:
        return self.model.forward(token_ids, mask, targets)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[CosineAnnealingLR]]:
        return [self.optimizer], [self.lr_scheduler]


    def training_step(self, batch, batch_idx):
        logits, loss = self.model.forward(**batch)

        # あまりにもモデルが学習するとtrain_lossがnan, infになるかもしれない
        # 学習を安定させるため、その場合は0.0に変更
        if isinstance(loss, torch.Tensor):
            if torch.isnan(loss) or torch.isinf(loss):
                loss = torch.tensor(0.0, device=loss.device)

            self.log(
                "train_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
            )

            return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self.model.forward(**batch)
        if isinstance(loss, torch.Tensor):
            self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        logits, loss = self.model.forward(**batch)
        if isinstance(loss, torch.Tensor):
            self.log("test_loss", loss)
            return loss


if __name__ == "__main__":
    gpt = GPT(
        vocab_size=256,
        n_embd=256,
        n_head=8,
        n_layer=12,
        d_ff=256,
        seq_len=100,
        dropout=0.5,
    )
    optimizer = IndependentAdamW(
        params=gpt.parameters(),
        lr=1e-3,
        weight_decay=0.03,
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        max_epochs=10,
        warmup_epochs=1,
        warmup_start_lr=1e-3,
        eta_min=1e-5,
    )
    model = GPTLightningModel(gpt, optimizer, lr_scheduler)
