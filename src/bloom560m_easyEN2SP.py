from pathlib import Path
import mlflow
import sqlite3
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
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


class MLFlowExperimentManager:
    TRACKING_SQL_PATH = "result/mlruns.db"
    EXPERIMENT_NAME = "bloom560m_easyEN2SP"
    RUN_NAME = "op-check-run"
    DESCRIPTION = """
    bloom560mで簡単なENからSPへの変換を学習する
    """

    def __init__(self):
        self._setup_tracking_sql()
        self._setup_experiment()
        self.run = self._start_run()

    def _setup_tracking_sql(self):
        db_path = Path(self.TRACKING_SQL_PATH)
        if not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
            print("make db_path")
        sqlite3.connect(db_path)
        mlflow.set_tracking_uri(f"sqlite:///{self.TRACKING_SQL_PATH}")

    def _setup_experiment(self):
        mlflow.enable_system_metrics_logging()
        mlflow.set_experiment(self.EXPERIMENT_NAME)

    def _start_run(self):
        mlflow.pytorch.autolog(
            log_models=False,  # Trueにすると最後の状態のモデルが保存される.今回は最後の状態ではなく,metricが良い状態のcheckpointを使用するためFalse
            checkpoint=False,#pytorch lighningのcheckpointを使用するのでFalse
        )
        return mlflow.start_run(
            run_name=self.RUN_NAME,
            description=self.DESCRIPTION,
        )

    def get_latest_checkpoint(self, model):
        return mlflow.pytorch.load_checkpoint(model, self.run.info.run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 勝手にmlrunsディレクトリが作成されて邪魔なため削除する
        # mlruns_dir = Path('mlruns')
        # if mlruns_dir.exists() and mlruns_dir.is_dir():
        #     shutil.rmtree(mlruns_dir)

        return self.run.__exit__(exc_type, exc_val, exc_tb)
