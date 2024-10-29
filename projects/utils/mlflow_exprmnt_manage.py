import sqlite3
from pathlib import Path

import mlflow


class MLFlowExperimentManager:
    TRACKING_SQL_PATH = "logs/mlruns.db"
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
            checkpoint=False,  # pytorch lighningのcheckpointを使用するのでFalse
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
