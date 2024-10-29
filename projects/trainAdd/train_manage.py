from pathlib import Path
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import uuid


# 工夫した点,Mlflowの依存をMLFlowExperimentMangerのみにした.パラメータを設定する部分はコンストラクタに直接代入して,型を間違えないように.mlflow.pytorch.autologを活用,チェックポイント生成など自動で行ってくれる.(lightningのcallbackでもcheckpoint作成はできるが,lightningで作ったあと,mlflowのartifactに登録しないといけないので,依存関係がゴチャゴチャすることを避ける.もっと詳細にカスタマイズしたいならlightningのcallbackを使用するのもあり.adamだとlossが発散してうまく学習しなかった.


def get_trainer(ckpt_path: str):
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.03,
        patience=10,
        verbose=True,
        mode="min",
    )

    if not Path(ckpt_path).exists():
        Path(ckpt_path).mkdir(parents=True, exist_ok=True)

    check_point_id = uuid.uuid4()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpt_path,
        filename=str(check_point_id) +"_{epoch}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        every_n_epochs=5,
    )

    trainer = Trainer(
        callbacks=[early_stopping, checkpoint_callback],
        logger=False,  # ログにはMLflowを使用するためFalse
        max_epochs=30,
        max_time={"hours": 3},
        accelerator="gpu",
        precision="bf16-mixed",
        devices=1,
    )
    return trainer, check_point_id

