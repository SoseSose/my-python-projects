import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from loguru import logger

from trainAdd.data_proc.data_make import 足し算ドリル


class 足し算ドリルDM(LightningDataModule):
    def __init__(self, batch_size: int, seq_len: int):
        super().__init__()
        self.batch_size = batch_size
        self.ds = 足し算ドリル(seq_len)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.dl = DataLoader(
            dataset=self.ds,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def train_dataloader(self):
        return self.dl

    def val_dataloader(self):
        return self.dl

    def test_dataloader(self):
        return self.dl

    def predict_dataloader(self):
        return self.dl

def テスト_batch_type():
    batch_size = 2
    seq_len = 32

    logger.info(f"batch_size: {batch_size}, seq_len: {seq_len}")
    dm = 足し算ドリルDM(batch_size=batch_size, seq_len=seq_len)
    logger.info(f"batch_size: {batch_size}, seq_len: {seq_len}")

    dm.setup("fit")
    dl = dm.train_dataloader()
    
    for batch in dl:
        assert batch["token_ids"].shape == (batch_size, seq_len), "token_ids shape mismatch"
        assert batch["mask"].shape == (batch_size, seq_len), "mask shape mismatch"
        assert batch["targets"].shape == (batch_size, seq_len), "targets shape mismatch"
        assert batch["token_ids"].dtype == torch.long, "token_ids dtype mismatch"
        assert batch["mask"].dtype == torch.bool, "mask dtype mismatch"
        assert batch["targets"].dtype == torch.long, "targets dtype mismatch"
        break

