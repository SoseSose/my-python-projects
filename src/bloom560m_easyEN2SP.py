from pathlib import Path
import mlflow
import sqlite3
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
import shutil
from torch.utils.data import DataLoader


#工夫した点,Mlflowの依存をMLFlowExperimentMangerのみにした.パラメータを設定する部分はコンストラクタに直接代入して,型を間違えないように.mlflow.pytorch.autologを活用,チェックポイント生成など自動で行ってくれる.(lightningのcallbackでもcheckpoint作成はできるが,lightningで作ったあと,mlflowのartifactに登録しないといけないので,依存関係がゴチャゴチャすることを避ける.もっと詳細にカスタマイズしたいならlightningのcallbackを使用するのもあり.adamだとlossが発散してうまく学習しなかった.

def get_trainer():
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.03,
        patience=5,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        callbacks=[early_stopping],
        logger=False,  # ログにはMLflowを使用するためFalse
        # logger=logger,
        enable_checkpointing=False,  # チェックポイントにはMLflowを使用するためFalse
        max_epochs=30,
        max_time={"hours": 3},
        accelerator="gpu",
        precision="bf16-mixed",
        devices=1,
    )
    return trainer

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
            log_models=False,  #Trueにすると最後の状態のモデルが保存される.今回は最後の状態ではなく,metricが良い状態のcheckpointを使用するためFalse
            checkpoint=True,
            checkpoint_monitor="val_loss",
            checkpoint_mode="min",
        )
        return mlflow.start_run(
            run_name=self.RUN_NAME,
            description=self.DESCRIPTION,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        #勝手にmlrunsディレクトリが作成されて邪魔なため削除する
        mlruns_dir = Path('mlruns')
        if mlruns_dir.exists() and mlruns_dir.is_dir():
            shutil.rmtree(mlruns_dir)
        
        return self.run.__exit__(exc_type, exc_val, exc_tb)
    

def mlflow_evaluator(model:LightningModule, dataloader:DataLoader):
    for data in dataloader:
        model.validation_step(data)
        mlflow.log()

    