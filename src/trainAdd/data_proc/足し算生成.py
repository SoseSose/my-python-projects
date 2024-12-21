from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import random
import shutil
from loguru import logger

@dataclass
class 二項足し算パラメータ:
    桁数1: int
    桁数2: int
    サンプル割合1: float 
    サンプル割合2: float

@dataclass
class 多項足し算パラメータ:
    項数: int
    サンプル割合: float

class 足し算生成器:
    def __init__(self):
        self.bs_tok = "<"  # bos_token
        self.es_tok = ">"  # eos_token

    def n桁の重複なしランダム整数生成(self, 桁数: int, サンプル割合: float) -> list[int]:
        unique_numbers = set()
        start = 10 ** (桁数 - 1)
        stop = 10**桁数 - 1
        sample_num = int((stop - start) * サンプル割合)
        
        if sample_num > 10000:
            raise ValueError(f"sample_numが大きすぎます: {sample_num}")

        while len(unique_numbers) < sample_num:
            num = random.randint(start, stop)
            unique_numbers.add(num)

        return list(unique_numbers)

    def 二項足し算文字列生成(self, パラメータ: 二項足し算パラメータ) -> str:
        digit1_sampled = self.n桁の重複なしランダム整数生成(パラメータ.桁数1, パラメータ.サンプル割合1)
        digit2_sampled = self.n桁の重複なしランダム整数生成(パラメータ.桁数2, パラメータ.サンプル割合2)

        sample_num = len(digit1_sampled) * len(digit2_sampled)
        if sample_num > 10000:
            raise ValueError("サンプル数が多すぎる")

        問題リスト = []
        for i in digit1_sampled:
            for j in digit2_sampled:
                問題リスト.append(f"{i}+{j}={self.bs_tok}{i+j}{self.es_tok}")
                問題リスト.append(f"{j}+{i}={self.bs_tok}{j+i}{self.es_tok}")  # 交換法則学習用

        return "\n".join(問題リスト) + "\n"

    def 多項足し算文字列生成(self, パラメータ: 多項足し算パラメータ) -> str:
        if パラメータ.項数 < 2:
            raise ValueError("項数は2以上である必要がある")

        randints = self.n桁の重複なしランダム整数生成(パラメータ.項数, パラメータ.サンプル割合)
        
        問題リスト = []
        for one_randint in randints:
            ans = 0
            式 = ""
            for i in str(one_randint):
                ans += int(i)
                式 += f"{i}+"
            式 = 式[:-1]  # 最後の+を削除
            問題リスト.append(f"{式}={self.bs_tok}{ans}{self.es_tok}")

        return "\n".join(問題リスト) + "\n"

class データセット生成器:
    def __init__(self, 生成器: 足し算生成器):
        self.生成器 = 生成器

    def _ディレクトリ初期化(self, path: Path):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def 二項足し算ファイル生成(self, dir_path: Path, パラメータリスト: List[二項足し算パラメータ]):
        self._ディレクトリ初期化(dir_path)
        for パラメータ in パラメータリスト:
            問題文字列 = self.生成器.二項足し算文字列生成(パラメータ)
            with open(dir_path / f"{パラメータ.桁数1}桁{パラメータ.桁数2}桁.txt", "w") as f:
                f.write(問題文字列)

    def 多項足し算ファイル生成(self, dir_path: Path, パラメータリスト: List[多項足し算パラメータ]):
        self._ディレクトリ初期化(dir_path)
        for パラメータ in パラメータリスト:
            問題文字列 = self.生成器.多項足し算文字列生成(パラメータ)
            with open(dir_path / f"{パラメータ.項数}項.txt", "w") as f:
                f.write(問題文字列)

def 桁数の汎化データセット生成(base_path: Path, train_params: List[二項足し算パラメータ], test_params: List[二項足し算パラメータ]):
    生成器 = 足し算生成器()
    データセット = データセット生成器(生成器)
    
    data_path = base_path / "桁数の汎化"
    データセット.二項足し算ファイル生成(data_path / "train", train_params)
    データセット.二項足し算ファイル生成(data_path / "test", test_params)

def 項数の汎化データセット生成(base_path: Path, train_params: List[多項足し算パラメータ], test_params: List[多項足し算パラメータ]):
    生成器 = 足し算生成器()
    データセット = データセット生成器(生成器)
    
    data_path = base_path / "項数の汎化"
    データセット.多項足し算ファイル生成(data_path / "train", train_params)
    データセット.多項足し算ファイル生成(data_path / "test", test_params)

def main():
    base_path = Path("dataset")
    
    # 桁数の内挿汎化
    train_params = [
        二項足し算パラメータ(1, 1, 1.0, 1.0),
        二項足し算パラメータ(2, 2, 0.3, 0.3),
        # ... 他のパラメータ
    ]
    test_params = [
        二項足し算パラメータ(3, 3, 0.1, 0.1),
        # ... 他のパラメータ
    ]
    桁数の汎化データセット生成(base_path / "桁数の内挿汎化", train_params, test_params)

    # 項数の内挿汎化
    train_params_多項 = [
        多項足し算パラメータ(2, 1.0),
        多項足し算パラメータ(4, 0.1),
        # ... 他のパラメータ
    ]
    test_params_多項 = [
        多項足し算パラメータ(3, 0.1),
        多項足し算パラメータ(5, 1e-4),
        # ... 他のパラメータ
    ]
    項数の汎化データセット生成(base_path / "項数の内挿汎化", train_params_多項, test_params_多項)

if __name__ == "__main__":
    main()
