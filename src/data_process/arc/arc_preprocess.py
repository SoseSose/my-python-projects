# %%
import json
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import List, Union

import numpy as np
import numpy.typing as npt
import pytest
import lightning as L
from torch.utils.data import DataLoader

from data_process.arc.const import ArcConst

MIN_COLOR_NUM = ArcConst().MIN_COLOR_NUM
MAX_COLOR_NUM = ArcConst().MAX_COLOR_NUM
MAX_SIZE = ArcConst().MAX_IMG_SIZE
VR_DELIM = ""
HR_DELIM = "\n"

class ArcImage:
    """
    ARCタスクの画像を表現するクラスです。

    属性:
    - img: 2次元リストとして保存された画像データ
    - x: 画像の幅
    - y: 画像の高さ

    メソッド:
    - to_str: 画像を文字列として表現
    - to_2d_list: 画像を2次元リストとして返す
    - to_np: 画像をNumPy配列として返す

    プロパティ:
    - to_np: 画像をNumPy配列として返す

    特殊メソッド:
    - __str__: 画像の文字列表現を返す
    - __array__: NumPy配列としての表現を返す
    - __eq__: 他のArcImageオブジェクトとの等価性を比較
    """

    def __init__(
        self, original_2d_list: Union[List[List[int]], npt.NDArray[np.int32]]
    ) -> None:
        """
        ArcImageオブジェクトを初期化します。

        パラメータ:
        - original_2d_list: 2次元リストまたはNumPy配列として表現された画像データ

        例外:
        - ValueError: 入力データが無効な場合（行の長さが不均一、サイズが大きすぎる、
                      要素が整数でない、色の値が範囲外）
        """
        for row in original_2d_list:
            if len(row) != len(original_2d_list[0]):
                raise ValueError("All rows must have the same length.")
        if len(original_2d_list) > MAX_SIZE or len(original_2d_list[0]) > MAX_SIZE:
            raise ValueError("The size of the image is too large.")

        for row in original_2d_list:
            for num in row:
                if not isinstance(num, int):
                    raise ValueError("All elements must be int.")
                if num < MIN_COLOR_NUM:
                    raise ValueError("All elements must be > 0")
                if num > MAX_COLOR_NUM:
                    raise ValueError("All elements must be < {}".format(MAX_COLOR_NUM))

        if isinstance(original_2d_list, list):
            self.img = original_2d_list
        else:
            self.img = original_2d_list.tolist()

        self.x = len(original_2d_list[0])
        self.y = len(original_2d_list)

    def to_str(self) -> str:
        """画像を文字列として表現します。"""
        char_two_d_list = [[str(one_num) for one_num in row] for row in self.img]
        # num to char
        row_joined_list = [VR_DELIM.join(row) for row in char_two_d_list]
        return HR_DELIM.join(row_joined_list)

    def to_2d_list(self) -> List[List[int]]:
        """画像を2次元リストとして返します。"""
        return self.img

    def __str__(self) -> str:
        """画像の文字列表現を返します。"""
        return self.to_str()

    @property
    def to_np(self) -> npt.NDArray[np.int32]:
        """画像をNumPy配列として返します。"""
        return np.array(self.img)

    def __array__(self):
        """NumPy配列としての表現を返します。"""
        return np.array(self.img)

    def __eq__(self, other: "ArcImage") -> bool:
        """
        他のArcImageオブジェクトとの等価性を比較します。

        パラメータ:
        - other: 比較対象のArcImageオブジェクト

        戻り値:
        - bool: 両者が等しい場合はTrue、そうでない場合はFalse
        """
        if self.to_np.shape != other.to_np.shape:
            return False
        else:
            return (self.to_np == other.to_np).all()


class TestArcImage:
    def get_test_image(self) -> List[List[int]]:
        return [[i for i in range(ArcConst().MIN_COLOR_NUM,ArcConst().MAX_COLOR_NUM+1)] for _ in range(6)]
    def test_init_normal(self):
        test_image = self.get_test_image()
        arc_image = ArcImage(test_image)
        assert arc_image.img == test_image

    def test_init_lager_value(self):
        test_image = self.get_test_image()
        test_image[0][0] = 12
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_init_smaller_value(self):
        test_image = self.get_test_image()
        test_image[0][0] = -1
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_init_not_int(self):
        test_image = self.get_test_image()
        test_image[0][0] = "0"  # type: ignore
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_init_not_same_length(self):
        test_image = self.get_test_image()
        test_image[0].append(1)
        with pytest.raises(ValueError):
            ArcImage(test_image)

    def test_to_string(self):
        test_image = self.get_test_image()
        arc_image = ArcImage(test_image)
        assert (
            arc_image.to_str()
            == "0123456789\n0123456789\n0123456789\n0123456789\n0123456789\n0123456789"
        )

    def test_to_2d_list(self):
        test_image = self.get_test_image()
        arc_image = ArcImage(test_image)
        assert arc_image.to_2d_list() == test_image

    def test_str(self):
        test_image = [[1, 2], [3, 4]]
        arc_image = ArcImage(test_image)
        assert str(arc_image) == "12\n34"
    
    def test_ArcImage_equivarent_return_True(self):
        test_image = [[1, 2], [3, 4]]
        arc_image = ArcImage(test_image)
        arc_image2 = ArcImage(test_image)
        assert arc_image == arc_image2
    
    def test_ArcImage_unequivarent_return_False(self):
        arc_image = ArcImage([[1, 2], [3, 4]])
        arc_image2 = ArcImage([[1, 2], [3, 5]])
        assert arc_image != arc_image2

    def test_ArcImage_eq_not_same_shape_return_False(self):
        arc_image = ArcImage([[1, 2], [3, 4]])
        arc_image2 = ArcImage([[1, 2, 2], [3, 4,4]])
        assert arc_image != arc_image2


@dataclass
class ArcInout:
    input: ArcImage
    output: ArcImage

    def __str__(self) -> str:
        # return f"input:\n{self.input}\noutput:\n{self.output}"
        return f"{self.input}\n->\n{self.output}"

    def __eq__(self, other: "ArcInout") -> bool:
        return self.input == other.input and self.output == other.output

def test_Arc_inout_input_not_same_return_false():
    test_image = [[1,2], [3,4]]
    arc_inout = ArcInout(ArcImage(test_image), ArcImage(test_image))
    arc_inout2 = ArcInout(ArcImage([[1,2],[3, 5]]), ArcImage(test_image))

    assert arc_inout != arc_inout2

def test_Arc_inout_output_not_same_return_false():
    test_image = [[1,2], [3,4]]
    arc_inout = ArcInout(ArcImage(test_image), ArcImage(test_image))
    arc_inout2 = ArcInout(ArcImage([[1,2],[3, 5]]), ArcImage(test_image))
    assert arc_inout != arc_inout2

def test_Arc_inout_input_same_return_true():
    test_image = [[1,2], [3,4]]
    arc_inout = ArcInout(ArcImage(test_image), ArcImage(test_image))
    arc_inout2 = ArcInout(ArcImage(test_image), ArcImage(test_image))

    assert arc_inout == arc_inout2

TRAIN_NAME = "train"
TEST_NAME = "test"

@dataclass
class ArcTask:
    """
    ARCタスクを表現するクラスです。

    属性:
    - train: 学習用の入出力ペアのリスト
    - test: テスト用の入出力ペア
    - candidate: 候補となる出力画像のリスト
    - name: タスクの名前（デフォルトは "no name"）

    プロパティ:
    - train_inputs: 学習用の入力画像のリスト
    - train_outputs: 学習用の出力画像のリスト
    - test_input: テスト用の入力画像
    - test_output: テスト用の出力画像
    - question: タスクの文字列表現（学習データとテスト入力を含む）

    メソッド:
    - to_str: タスクの文字列表現を生成
    """

    train: list[ArcInout]
    test: ArcInout
    candidate: list[ArcImage]
    name: str = "no name"

    @property
    def train_inputs(self) -> list[ArcImage]:
        return [inout.input for inout in self.train]

    @property
    def train_outputs(self) -> list[ArcImage]:
        return [inout.output for inout in self.train]

    @property
    def test_input(self) -> Union[ArcImage, list[ArcImage]]:
        if isinstance(self.test, list):
            return [inout.input for inout in self.test]
        else:
            return self.test.input

    @property
    def test_output(self) -> Union[ArcImage, list[ArcImage]]:
        if isinstance(self.test, list):
            return [inout.output for inout in self.test]
        else:
            return self.test.output

    def to_str(
        self,
        show_test_out: bool = False,
        show_candidata: bool = False,
    ) -> str:
        rslt = ""

        for i, inout in enumerate(self.train):
            rslt += f"-{TRAIN_NAME}{i}-\n{inout}\n\n"

        if show_candidata:
            for i, one_candidate in enumerate(self.candidate):
                rslt += f"-candidate{i}-\n{one_candidate}\n\n"

        rslt += f"-{TEST_NAME}-\n"
        rslt += f"{self.test_input}\n->"

        if show_test_out:
            rslt += f"\n{self.test_output}"

        return rslt
    
    @property
    def question(self) -> str:
        return self.to_str()
    
    def __str__(self) -> str:
        return self.to_str()

    def __eq__(self, other: "ArcTask") -> bool:
        
        return (
            self.train == other.train
            and self.test == other.test
            and self.candidate == other.candidate
        )

class TestArcTask:
    def test_train_inputs(self):
        true_arc_image = ArcImage([[1, 2], [3, 4]])
        false_arc_image = ArcImage([[5, 6], [7, 8]])
        arc_inout = ArcInout(true_arc_image, false_arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[false_arc_image],
        )
        assert arc_task.train_inputs == [true_arc_image, true_arc_image]

    def test_train_outputs(self):
        true_arc_image = ArcImage([[1, 2], [3, 4]])
        false_arc_image = ArcImage([[5, 6], [7, 8]])
        arc_inout = ArcInout(true_arc_image, false_arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[false_arc_image],
        )
        assert arc_task.train_outputs == [false_arc_image, false_arc_image]

    def test_test_input(self):
        true_arc_image = ArcImage([[1, 2], [3, 4]])
        false_arc_image = ArcImage([[5, 6], [7, 8]])
        arc_inout = ArcInout(true_arc_image, false_arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[false_arc_image],
        )
        assert arc_task.test_input == true_arc_image

    def test_test_output(self):
        true_arc_image = ArcImage([[1, 2], [3, 4]])
        false_arc_image = ArcImage([[5, 6], [7, 8]])
        arc_inout = ArcInout(true_arc_image, false_arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[false_arc_image],
        )
        assert arc_task.test_output == false_arc_image

    def test_str(self):
        test_image = [[1, 2], [3, 4]]
        arc_image = ArcImage(test_image)
        arc_inout = ArcInout(arc_image, arc_image)
        arc_task = ArcTask(
            train=[arc_inout, arc_inout],
            test=arc_inout,
            candidate=[arc_image],
        )

        rslt = arc_task.to_str(show_candidata=True, show_test_out=True)

        need_str = """\
        -train0-
        12
        34
        ->
        12
        34
        
        -train1-
        12
        34
        ->
        12
        34

        -candidate0-
        12
        34
        
        -test-
        12
        34
        ->
        12
        34
        """

        need_str = textwrap.dedent(need_str).strip()
        assert rslt == need_str


def test_ArcTask_eq_same_return_True():
    test_image = [[1, 2], [3, 4]]
    arc_image = ArcImage(test_image)
    arc_inout = ArcInout(arc_image, arc_image)
    arc_task = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[arc_image],
    )
    arc_task2 = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[arc_image],
    )
    assert arc_task == arc_task2


def test_ArcTask_eq_not_same_train_return_False():
    test_image = [[1, 2], [3, 4]]
    arc_image = ArcImage(test_image)
    arc_inout = ArcInout(arc_image, arc_image)
    arc_inout2 = ArcInout(arc_image, ArcImage([[1, 2], [3, 5]]))
    arc_task = ArcTask(
        train=[arc_inout, arc_inout2],
        test=arc_inout,
        candidate=[arc_image],
    )
    arc_task2 = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[arc_image],
    )
    assert arc_task != arc_task2

def test_ArcTask_eq_not_same_test_return_False():
    test_image = [[1, 2], [3, 4]]
    arc_image = ArcImage(test_image)
    arc_inout = ArcInout(arc_image, arc_image)
    arc_inout2 = ArcInout(arc_image, ArcImage([[1, 2], [3, 5]]))
    arc_task = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[arc_image],
    )
    arc_task2 = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout2,
        candidate=[ArcImage([[1, 2], [3, 5]])],
    )
    assert arc_task != arc_task2

def test_ArcTask_eq_not_same_candidate_return_False():
    test_image = [[1, 2], [3, 4]]
    arc_image = ArcImage(test_image)
    arc_inout = ArcInout(arc_image, arc_image)
    arc_task = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[arc_image],
    )
    arc_task2 = ArcTask(
        train=[arc_inout, arc_inout],
        test=arc_inout,
        candidate=[ArcImage([[1, 2], [3, 5]])],
    )
    assert arc_task != arc_task2

class ArcTaskSet:
    """
    ARCタスクのセットを表現します。

    メソッド:
    - task_json_to_arc_task(task): JSON形式のタスクをArcTaskオブジェクトに変換します。
    - path_to_arc_task(data_path): JSONタスクファイルを含むディレクトリへのパスをArcTaskオブジェクトのリストに変換します。
    """

    def _task_json_to_arc_task(self, task, name: str):
        """
        JSON形式のタスクをArcTaskオブジェクトに変換します。

        パラメータ:
        - task: JSON形式のタスクを表す辞書。

        戻り値:
        - 変換されたタスクを表すArcTaskオブジェクト。
        """

        train = []
        for inout in task["train"]:
            input = ArcImage(inout["input"])
            output = ArcImage(inout["output"])
            train.append(ArcInout(input, output))

        test = task["test"]
        true_test_in = test[0]["input"]
        true_test_out = test[0]["output"]

        true_test_inout = ArcInout(ArcImage(true_test_in), ArcImage(true_test_out))

        candidate = []
        for two_cand in test[1 : ArcConst().FAKE_NUM + 1]:
            candidate.append(ArcImage(two_cand["input"]))
            candidate.append(ArcImage(two_cand["output"]))

        return ArcTask(train, true_test_inout, candidate, name)

    def path_to_arc_task(self, data_path: str) -> List[ArcTask]:
        """
        JSONタスクファイルを含むディレクトリへのパスをArcTaskオブジェクトのリストに変換します。

        パラメータ:
        - data_path: JSONタスクファイルを含むディレクトリへのパスを表すPathオブジェクト。

        戻り値:
        - 変換されたタスクを表すArcTaskオブジェクトのリスト。
        """
        tasks = []
        for task_file in Path(data_path).glob("*.json"):
            with task_file.open() as f:
                task = json.load(f)
                tasks.append(self._task_json_to_arc_task(task, task_file.stem))

        return tasks


def str_to_arc_image(string: str) -> ArcImage:
    """
    文字列表現を ArcImage オブジェクトに変換します。

    パラメータ:
    - string: 2次元リストの文字列表現。

    戻り値:
    - 変換された ArcImage オブジェクト。
    """
    two_d_list = string.split("\n")
    two_d_list = [list(row) for row in two_d_list]
    two_d_list = [[int(num) for num in row] for row in two_d_list]
    two_d_list = ArcImage(two_d_list)
    return two_d_list


class TestArcTaskSet:
    def test_op_path_to_arc_task(self):
        arc_task_set = ArcTaskSet()
        _ = arc_task_set.path_to_arc_task("data/training")

class ArcTaskDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.arc_task_set = ArcTaskSet()
        self.train_tasks = []
        self.test_tasks = []

    def prepare_data(self):
        train_tasks = self.arc_task_set.path_to_arc_task(self.data_path+"/training")
        self.train_tasks = [{"input":str(task.question), "output":str(task.test_output)} for task in train_tasks]

        eval_tasks = self.arc_task_set.path_to_arc_task(self.data_path+"/evaluation")
        self.test_tasks = [{"input":str(task.question), "output":str(task.test_output)} for task in eval_tasks]

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.train_tasks
            self.val_dataset = self.test_tasks  

        if stage == 'test' or stage is None:
            self.test_dataset = self.test_tasks

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
