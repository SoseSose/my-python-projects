import random
from dataclasses import dataclass
from typing import Optional



@dataclass
class ArcConst:
    # COLOR: tuple[int] = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    MIN_COLOR_NUM: int = 0
    MAX_COLOR_NUM: int = 9
    PAD_COLOR: int = 10  # 10はPADの色
    MAX_TRAIN_SIZE: int = 10
    MAX_IMG_SIZE: int = 30
    MIN_IMG_SIZE: int = 1

    FAKE_NUM = 8  # FAKEありのデータセットを作ったので,1つのタスクに含まれるFAKEの数


class ArcColor:
    def __init__(self):
        self.cand = [
            i for i in range(ArcConst.MIN_COLOR_NUM, ArcConst.MAX_COLOR_NUM)
        ]

    def pick_random_unused(self, index: Optional[int] = None) -> int:
        if index is not None:
            if index < 0 or index >= len(self.cand):
                raise Exception(f"index must be 0 <= index < {len(self.cand)}")
            picked = self.cand.pop(index)
        else:
            picked = self.cand.pop(random.randint(0, len(self.cand) - 1))
        return picked

import pytest

def test_color_pick():
    for _ in range(1000):
        color = ArcColor()
        first_picked_color = color.pick_random_unused(index=None)
        second_picked_color = color.pick_random_unused(index=None)
        assert first_picked_color != second_picked_color


def test_all_color_pick():
    color = ArcColor()
    for i in range(ArcConst.MAX_COLOR_NUM):
        picked_color = color.pick_random_unused(index=0)
        assert picked_color == i

    # it should be error
    color = ArcColor()
    with pytest.raises(Exception) as e:
        picked_color = color.pick_random_unused(index=-1)
    assert str(e.value) == f"index must be 0 <= index < {ArcConst.MAX_COLOR_NUM}"

    # it should not be
    color = ArcColor()
    picked_color = color.pick_random_unused(index=ArcConst.MAX_COLOR_NUM - 1)

    # it should be error
    color = ArcColor()
    with pytest.raises(Exception) as e:
        picked_color = color.pick_random_unused(index=ArcConst.MAX_COLOR_NUM)
    assert str(e.value) == f"index must be 0 <= index < {ArcConst.MAX_COLOR_NUM}"
