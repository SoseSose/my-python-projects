from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data_proc.data_make import 足し算ドリル


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

if __name__ == "__main__":
    dm = 足し算ドリルDM(batch_size=2, seq_len=32)
    dm.setup("fit")
    dl = dm.train_dataloader()
    for batch in dl:
        print(batch["token_ids"].shape)
        print(batch["mask"].shape)
        print(batch["targets"].shape)
        break
