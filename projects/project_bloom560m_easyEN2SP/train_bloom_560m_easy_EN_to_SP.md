# 英語からスペイン語への簡単な翻訳モデルの学習と記録

このガイドでは、MLflow、PyTorch Lightning、Hugging Faceを使用して、英語からスペイン語への簡単な翻訳モデルを学習し、その過程を記録する方法について説明します。

## 使用するライブラリ

- **MLflow**: 機械学習の実験を管理するためのオープンソースプラットフォーム。
- **PyTorch Lightning**: PyTorchのコードを簡潔にし、トレーニングループを管理するためのフレームワーク。
- **Hugging Face**: トランスフォーマーモデルのライブラリ。

## プロジェクトの構成

以下のファイルを使用します：

- `src/train_bloom_560m.py`: モデルのトレーニングスクリプト。
- `src/config/bloom560m_easyEN2SP.py`: トレーナーとMLflowの設定。
- `src/architectures/bloom_560m.py`: モデルのアーキテクチャ。
- `src/data_processing/easy_ds_EN_to_SP.py`: データ処理とデータモジュール。

## トレーニングスクリプト

以下は、`src/train_bloom_560m.py`の内容です：

```python
import utils
from architectures.bloom_560m import Bloom560m
from config.bloom560m_easyEN2SP import get_trainer, MLFlowExperimentManager
from data_processing.easy_ds_EN_to_SP import EasyEnToSpDM

utils.ini_setting()

def run_training():
    model = Bloom560m("D:/models", learning_rate=0.001)
    data_module = EasyEnToSpDM(tokenizer=model.tokenizer, batch_size=1)
    trainer = get_trainer()

    with MLFlowExperimentManager():
        trainer.fit(model=model, datamodule=data_module)
        trainer.test(model=model, datamodule=data_module)
        predictions = trainer.predict(model=model, datamodule=data_module, return_predictions=True)

    display_predictions(predictions)

def display_predictions(predictions):
    for prediction in predictions:
        print(prediction)

if __name__ == "__main__":
    run_training()
```

## トレーナーとMLflowの設定

以下は、`src/config/bloom560m_easyEN2SP.py`の内容です：

```python
import os
from pathlib import Path
import mlflow
import sqlite3
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

def get_trainer():
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.03,
        patience=10,
        verbose=True,
        mode="min",
    )

    trainer = Trainer(
        callbacks=[early_stopping],
        logger=False,
        enable_checkpointing=False,
        max_epochs=30,
        max_time={"hours": 3},
        accelerator="gpu",
        precision="bf16-mixed",
        devices=1,
    )
    return trainer

class MLFlowExperimentManager:
    def __init__(self):
        tracking_uri: str = "result/mlruns.db"
        experiment_name: str = "bloom560m_easyEN2SP"
        run_name: str = "op-check-run"
        description: str = """
        bloom560mで簡単なENからSPへの変換を学習する
        """

        db_path = Path(tracking_uri)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(db_path)

        mlflow.enable_system_metrics_logging()
        mlflow.set_tracking_uri(f"sqlite:///{tracking_uri}")
        mlflow.set_experiment(experiment_name)

        mlflow.pytorch.autolog(
            log_models=False,
            checkpoint=True,
            checkpoint_metric="val_loss",
            checkpoint_mode="min",
        )
        self.run = mlflow.start_run(
            run_name=run_name,
            description=description,
        )

    def __enter__(self):
        return self.run.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.run.__exit__(exc_type, exc_val, exc_tb)

