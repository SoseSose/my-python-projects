# %%
import json
import os
import sqlite3
import mlflow
import pandas as pd
from tqdm import tqdm
from data_processing.arc.arc_preprocess import ArcTaskSet, ArcTask,str_to_arc_image, ArcImage
from architectures.phi2 import BaseModel
from arc_visualize import plot_task
from mlflow.entities import ViewType
from pathlib import Path

DB_PATH = "result/mlruns.db"
ARTIFACT_LOCATION = "result/artifacts"
Phi2_OUTPUT_FILE = "phi-2 select.json"


class MlflowRapper:
    input_identifier = "input idetifer"
    question = "question"
    answer = "answer"
    token_num = "token num"

    def __init__(self, experiment_name) -> None:
        self.experiment_name = experiment_name
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        sqlite3.connect(DB_PATH)

        self.tracking_uri = f"sqlite:///{DB_PATH}"
        mlflow.set_tracking_uri(self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name, artifact_location=ARTIFACT_LOCATION
            )
        else:
            experiment_id = experiment.experiment_id

        self.experiment_id = experiment_id

    def evaluate_n_log(self, ds: list[ArcTask], model: BaseModel, train_or_eval):
        columns = [
            self.input_identifier,
            self.question,
            self.answer,
            self.token_num,
        ]
        data = {column: [] for column in columns}

        df = pd.DataFrame(data)

        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            mlflow.set_tag("train or Eval", train_or_eval)

            for i, data in tqdm(enumerate(ds)):
                input_identifier = data.name
                question = data.to_str()
                output, token_num = model.get_token_num_and_answer(question)
                df.loc[i] = [input_identifier, question, output, token_num]

            mlflow.log_table(df, Phi2_OUTPUT_FILE)
            run_id = run.info.run_id
            # dataframeのindexは保存されない。

        return run_id

    def catch_run_location(self, run_id: str) -> str:
        mlflow.set_tracking_uri(self.tracking_uri)
        runs = mlflow.search_runs(
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            output_format="list",
            experiment_names=[self.experiment_name],
        )

        for run in runs:
            if run.info.run_id == run_id:
                location = mlflow.artifacts.download_artifacts(run.info.artifact_uri)
                return location

    def gather_collect(self, df, ds):
        name_key_ds = {data.name: data for data in ds}

        collect_ans = {}
        for _, row in df.iterrows():
            name = row[self.input_identifier]
            data = name_key_ds[name]

            correct_ans = data.test_output
            if isinstance(correct_ans, ArcImage):
                correct_ans = correct_ans.to_str()

            model_answer = row[self.answer].split("\n\n")[0]

            if correct_ans == model_answer:
                collect_ans[name] = str_to_arc_image(model_answer)
        return collect_ans

    def get_collect_name_and_ans(self, run_id: str, ds: list[ArcTask]) -> dict:
        location = self.catch_run_location(run_id)

        with open(os.path.join(location, Phi2_OUTPUT_FILE)) as f:
            dic = json.load(f)
        df = pd.DataFrame(dic["data"], columns=dic["columns"])

        return self.gather_collect(df, ds)

    def calculate_num_collect(self, ds: list[ArcTask], model: BaseModel):
        analyze_id = self.evaluate_n_log(ds, model, "evaluation")
        collect_ans = self.get_collect_name_and_ans(analyze_id, ds)
        num_collect = len(collect_ans)
        return num_collect


def visualize(task: ArcTask, model_answer, file_path: str):
    plot_task(
        train_inputs=task.train_inputs,
        train_outputs=task.train_outputs,
        test_inout=[task.test_input, task.test_output],
        model_answer=model_answer,
        save_path=file_path,
    )


import pytest


@pytest.mark.skip(reason="this test takes too long")
def test_evalate_n_log(tmp_path: Path):
    mock = MockModel()

    os.chdir(tmp_path)
    # train_or_eval = "training"
    train_or_eval = "evaluation"
    experiment_name = "test"
    ds = ArcTaskSet().path_to_arc_task("data/" + train_or_eval)
    mlflow_rapper = MlflowRapper(experiment_name)

    run_id = mlflow_rapper.evaluate_n_log(ds, mock, train_or_eval)

    assert run_id is not None



import random
class MockModel(BaseModel):
    def __init__(self, ds:list[ArcTask], expected_collect_num:int) -> None:
        super().__init__()

        self.ques_and_ans = {data.question():data.test_output.to_str() for data in ds[:expected_collect_num]}

    def build(self):
        pass

    def get_token_num_and_answer(self, question:str)->tuple[str, int]:
        if question in self.ques_and_ans:
            answer = self.ques_and_ans[question]
        else:
            answer = "dummy"
        return answer, 0
    
    def get_answers(self, ds:list[ArcTask])->list[str]:
        answers = []

        for data in ds:
            question = data.question()
            answer = self.get_token_num_and_answer(question)[0]
            answers.append(answer)
        return answers

import random
def test_MockModel():
    ds = ArcTaskSet().path_to_arc_task("data/evaluation")
    expected_collect_num = random.randint(0, len(ds))

    collect_answers = [data.test_output.to_str() for data in ds]
    model = MockModel(ds, expected_collect_num=expected_collect_num)

    answers = model.get_answers(ds)

    collect_num = 0
    for ans, collect_ans in zip(answers, collect_answers):
        if ans == collect_ans:
            collect_num += 1

    assert expected_collect_num == collect_num


def test_calculate_num_collect(tmp_path: Path):

    ds = ArcTaskSet().path_to_arc_task("data/evaluation")
    expected_collect_num = random.randint(0, len(ds))
    mock = MockModel(expected_collect_num=expected_collect_num, ds=ds)
    os.chdir(tmp_path)

    experiment_name = "test"
    mlflow_rapper = MlflowRapper(experiment_name)

    num_collect = mlflow_rapper.calculate_num_collect(ds, mock)

    assert num_collect == expected_collect_num


