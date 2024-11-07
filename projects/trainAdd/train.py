import torch
from data_proc.datamodule import 足し算ドリルDM
from model.lightning_model import GPTLightningModel
from model.gpt import GPT
from model.modules import CosineAnnealingLR, IndependentAdamW
from my_utils.mlflow_expriment_manage import MLFlowExperimentManager
from train_manage import get_trainer
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 1なら同期実行されるのでデバッグがしやすい。


def main():
    dm = 足し算ドリルDM(batch_size=4, seq_len=16)
    vocab_size = dm.ds.tokenizer.vocab_size
    gpt = GPT(
        vocab_size=vocab_size,
        n_embd=128,
        n_head=8,
        n_layer=3,
        d_ff=128,
        seq_len=dm.ds.seq_len,
        dropout=0.1,
    )
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        warmup_epochs=10,
        max_epochs=500,
        warmup_start_lr=1e-8,
        eta_min=1e-5,
        last_epoch=-1,
    )

    model = GPTLightningModel(
        gpt=gpt,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer, checkpoint_id = get_trainer("logs/checkpoints")

    manager = MLFlowExperimentManager()
    with manager:
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
