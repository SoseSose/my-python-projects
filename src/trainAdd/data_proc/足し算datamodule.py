from pathlib import Path
from typing import Optional, List
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import torch
from loguru import logger

from trainAdd.data_proc.足し算生成 import (
    足し算生成器,
    データセット生成器,
    二項足し算パラメータ,
    多項足し算パラメータ,
)

class 足し算データセット(Dataset):
    def __init__(self, データパス: Path, seq_len: int):
        self.seq_len = seq_len
        self.問題リスト = []
        
        for ファイルパス in データパス.glob("*.txt"):
            with open(ファイルパス, "r") as f:
                self.問題リスト.extend(f.read().strip().split("\n"))
        
        logger.info(f"読み込んだ問題数: {len(self.問題リスト)}")

    def __len__(self):
        return len(self.問題リスト)

    def __getitem__(self, idx):
        問題 = self.問題リスト[idx]
        token_ids = [ord(c) for c in 問題]
        
        if len(token_ids) < self.seq_len:
            token_ids = token_ids + [0] * (self.seq_len - len(token_ids))
        else:
            token_ids = token_ids[:self.seq_len]
        
        mask = [1 if tid != 0 else 0 for tid in token_ids]
        targets = token_ids.copy()
        
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "targets": torch.tensor(targets, dtype=torch.long),
        }

class 足し算DataModule(LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        batch_size: int = 32,
        seq_len: int = 128,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.train_dataset = 足し算データセット(self.data_path / "train", self.seq_len)
        self.val_dataset = 足し算データセット(self.data_path / "test", self.seq_len)
        self.test_dataset = 足し算データセット(self.data_path / "test", self.seq_len)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        return self.test_dataloader()

def テスト_batch_type():
    """各DataModuleのバッチの型をテストする関数"""
    base_path = Path("dataset")
    batch_size = 2
    seq_len = 32

    logger.info("桁数の内挿汎化のテスト")
    dm = 足し算DataModule(
        data_path=base_path/"桁数の内挿汎化",
        batch_size=batch_size,
        seq_len=seq_len,
    )

    dm.prepare_data()
    dm.setup("fit")
    dl = dm.train_dataloader()
    
    for batch in dl:
        assert batch["token_ids"].shape == (batch_size, seq_len), "token_ids shape mismatch"
        assert batch["mask"].shape == (batch_size, seq_len), "mask shape mismatch"
        assert batch["targets"].shape == (batch_size, seq_len), "targets shape mismatch"
        break


if __name__ == "__main__":
    テスト_batch_type()
