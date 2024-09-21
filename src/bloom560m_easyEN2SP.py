from pathlib import Path
import mlflow
import sqlite3
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# 工夫した点,Mlflowの依存をMLFlowExperimentMangerのみにした.パラメータを設定する部分はコンストラクタに直接代入して,型を間違えないように.mlflow.pytorch.autologを活用,チェックポイント生成など自動で行ってくれる.(lightningのcallbackでもcheckpoint作成はできるが,lightningで作ったあと,mlflowのartifactに登録しないといけないので,依存関係がゴチャゴチャすることを避ける.もっと詳細にカスタマイズしたいならlightningのcallbackを使用するのもあり.adamだとlossが発散してうまく学習しなかった.


def get_mlflow_logger(artifact_dir:str):
    db_path = Path("result/mlruns.db")
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        print("make db_path")

    mlf_logger = MLFlowLogger(
        experiment_name="bloom560m_easyEN2SP",
        run_name="op-check-run",
        tracking_uri=f"sqlite:///{db_path}",
        log_model=True,
        artifact_dir=artifact_dir,
    )

    #システムメトリックが取れなくなっている。取る時はtest_step中にnvmlを使って取るようにするとか

    return mlf_logger


def get_trainer():
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.03,
        patience=10,
        verbose=True,
        mode="min",
    )

    check_point_dir = "result/checkpoints"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=check_point_dir,
        filename="bloom560m_easyEN2SP-{epoch}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    mlf_logger = get_mlflow_logger(artifact_dir=check_point_dir)
    # mlflow.pytorch.autologはtest時にlogされないため、lightningのmlf_loggerを使ってlogする

    trainer = Trainer(
        callbacks=[early_stopping, checkpoint_callback],
        logger=mlf_logger,
        max_epochs=30,
        max_time={"hours": 3},
        accelerator="gpu",
        precision="bf16-mixed",
        devices=1,
    )
    return trainer
