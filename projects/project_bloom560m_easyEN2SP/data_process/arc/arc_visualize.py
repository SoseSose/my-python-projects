# %%
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Union, Optional
from pathlib import Path

from data_process.arc.const import ArcConst

MIN_COLOR_NUM = ArcConst.MIN_COLOR_NUM
MAX_COLOR_NUM = ArcConst.MAX_COLOR_NUM
MAX_IMG_SIZE = ArcConst.MAX_IMG_SIZE

#!　将来的にarcのメタデータを入手できるように
#! figやgsの処理が甘い。もっとシンプルに書けるはず。


def plot_one(ax, input, show_num=False):
    """画像を一つ表示する関数

    Args:
        ax (_type_): _description_
        input (_type_): _description_
        show_num (bool, optional):動作検証用
    """
    color_list = [
        "#FFFFFF",
        "#000000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#FFD700",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
        "#808080",
    ]
    cmap = colors.ListedColormap(color_list)
    norm = colors.Normalize(MIN_COLOR_NUM, MAX_COLOR_NUM)
    ax.imshow(input, cmap=cmap, norm=norm)

    shape = input.shape

    ax.tick_params(length=0)
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + shape[0])])
    ax.set_xticks([x - 0.5 for x in range(1 + shape[1])])

    if show_num:
        for x in range(shape[1]):
            for y in range(shape[0]):
                plt.text(x, y, str(input[y, x]), size=6)

    ax.set_xticklabels([])
    ax.set_yticklabels([])


def input_format(
    input: Union[np.ndarray, list],
    pad=None,
):
    """
    Args:
        input (Union[np.ndarray, list]): [N, H, W, C]または[N, H, W]であるarc_image
        pad (Union[None, list], optional): None or [pad height, w_height]. Defaults to None. もし値がある時はpad_height, pad_widthの値までpaddingされる。今は使っていない。
    """

    if type(input) != np.ndarray:
        input = [np.array(one_input) for one_input in input]

    input_dim = len(input[0].shape) + 1
    if not 3 <= input_dim <= 4:
        raise ValueError(
            "input must be [N, H, W, C] or [N, H, W],but got  dim:{}".format(input_dim)
        )
    elif input_dim == 4:
        input = np.argmax(input, axis=-1)

    if pad is not None:
        rslt = []
        for val in input:
            shape = val.shape
            val = np.pad(
                val,
                [[0, pad[0] - shape[0]], [0, pad[1] - shape[1]]],
                constant_values=0,
            )
            rslt.append(val)
    else:
        rslt = input
    return rslt


def plot_some(
    input: Union[np.ndarray, list],
    title: Union[str, list[str]],
    fig,
    gs,
    index,
    vis_len,
    show_num=False,
):
    """
    いくつかまとめて画像を表示する関数
    Args:
        input (np.ndarray):[N,H,W,C] or [N,H,W]のnp.ndarrayかlist, [N,H,W,C]の場合は最終次元でargmaxされる。
        title (str): 画像の上部に表示するテキスト
        pad (bool, optional): None or [pad height, w_height]. Defaults to None. もし値がある時はpad_height, pad_widthの値までpaddingされる。
        save_file_name (_type_, optional):None or str. Defaults to None. 画像を保存する場合は保存場所と名前を指定する。
        show_num (bool): Defaults to False. 画像に数字を表示するかどうか。
    """
    input = input_format(input)
    input_len = len(input)
    w = vis_len // (input_len + 1)
    for i, val in enumerate(input):
        if isinstance(title, str):
            retitle = title + ":" + str(i)
        elif isinstance(title, list):
            retitle = title[i]
        else:
            raise ValueError("title must be str or list")

        ax = fig.add_subplot(gs[index, i * w : (i + 1) * w])
        ax.set_title(retitle, fontdict={"fontsize": 50})
        plot_one(ax, val, show_num)

    return fig


def plot_task(
    train_inputs,
    train_outputs,
    test_inout,
    candidate=None,
    model_answer=None,
    save_path=None,
):
    """
    taskレベルで画像をまとめて表示する関数
    Args:
        train : [N, H, W, C]または[N, H, W]であるarc_image
        test_inout: [test_input, test_output]の形式でありtest_input, test_outputは[H, W, C]または[H , W]であるarc_image
        candidate :[N, H, W, C]または[N, H, W]. Defaults to None. Noneの場合は表示しない。
        model_answer : [H, W, C]であるarc_image. Defaults to None. Noneの場合は表示しない。
        save_path: Defaults to None. Noneの場合は表示する。strの場合はその名前で保存する。
    """
    fig = plt.figure(figsize=(MAX_IMG_SIZE, MAX_IMG_SIZE))

    height = 3
    if not isinstance(candidate, type(None)):
        height += 1
    if not isinstance(model_answer, type(None)):
        height += 1

    vis_len = 100
    gs = fig.add_gridspec(height, vis_len)

    plot_some(train_inputs, "train in", fig, gs, 0, vis_len)
    plot_some(train_outputs, "train out", fig, gs, 1, vis_len)

    plot_some(test_inout, ["test in", "test out"], fig, gs, 2, vis_len)

    gs_index = 2

    if not isinstance(candidate, type(None)):
        gs_index += 1
        plot_some(candidate, "candidate", fig, gs, gs_index, vis_len)

    if not isinstance(model_answer, type(None)):
        gs_index += 1
        plot_some([model_answer], "model answer", fig, gs, gs_index, vis_len)

    if save_path is not None:
        plt.show()
    elif isinstance(save_path, str):
        fig.savefig(save_path)
        plt.close()


import pytest


@pytest.mark.skip
def test_plot_task():
    test_image = np.tile(np.arange(10), (2, 6, 1))
    file_name = "test.png"

    plot_task(
        train_inputs=test_image,
        train_outputs=test_image,
        test_inout=[test_image[0], test_image[1]],
        # candidate=test_image,
        model_answer=test_image[1],
        save_path=file_name,
    )
    Path(file_name).unlink()


@pytest.mark.skip
class TestPlotSome:
    test_image = np.tile(np.arange(MAX_COLOR_NUM + 2), (2, 6, 1))

    def test_normal_plot(self):
        plot_some(self.test_image, "original", show_num=True)

    @pytest.mark.parametrize("fold", [2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_fold(self, fold):
        plot_some(self.test_image + 4, "original", show_num=True, fold=fold)

    def test_padded_plot(self):
        plot_some(self.test_image, "padded image", pad=(MAX_IMG_SIZE, MAX_IMG_SIZE))

    def test_one_hot_plot(self):
        def one_hot(x: np.ndarray, depth):
            return np.identity(depth)[x]

        one_hot_img = one_hot(self.test_image, MAX_COLOR_NUM + 2)
        plot_some(one_hot_img, "image", pad=(MAX_IMG_SIZE, MAX_IMG_SIZE))

    def test_save_path(self):
        file_name = "test.png"

        plot_some(
            self.test_image,
            "test save",
            pad=(MAX_IMG_SIZE, MAX_IMG_SIZE),
        )
        file_path = Path(file_name)
        if not file_path.exists():
            raise ValueError("file not saved")
        else:
            file_path.unlink()
