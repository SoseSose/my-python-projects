import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

# 1文字が1トークンにencodeされるとして実装している
# もしbyteベースのencodeにする際は要変更

bs_tok = "<"  # bos_token
es_tok = ">"  # eos_token


class Tokenizer:
    pad_token = "□"

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.str_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_str = {i: ch for i, ch in enumerate(self.chars)}

        # pad_tokenを追加
        vocab_size = len(self.chars)
        self.str_to_int.update({self.pad_token: vocab_size})
        self.int_to_str.update({vocab_size: self.pad_token})
        self.vocab_size = vocab_size + 1

    def encode(self, val: str) -> torch.Tensor:
        ints = [self.str_to_int[c] for c in val]
        return torch.tensor(ints, dtype=torch.long)

    def make_causal_data(self, token_ids: Tensor, seq_len: int) -> dict:
        original_len = len(token_ids)

        # 最後のtokenを除いてseq_lenまでpadする
        x = token_ids[:-1]
        pad_id = self.str_to_int[self.pad_token]
        x = F.pad(x, (0, seq_len - original_len + 1), "constant", pad_id)

        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[: original_len - 1] = 1

        target = torch.full((seq_len,), -100, dtype=torch.long)
        target[original_len - 1] = token_ids[-1]

        return {"token_ids": x, "mask": mask, "targets": target}

    def decode(self, val: torch.Tensor) -> str:
        ints = val.tolist()
        return "".join([self.int_to_str[i] for i in ints])


def テスト_tokenizer():
    from pprint import pprint

    text = "123456789"
    tokenizer = Tokenizer(text)
    encoded = tokenizer.encode(text)
    pprint(encoded)
    causal_data = tokenizer.make_causal_data(encoded, 13)
    pprint(causal_data)

    assert len(causal_data["token_ids"]) == len(causal_data["mask"]) == len(causal_data["target"])


def n桁のランダム整数生成(n_digit: int, sample_rate: float) -> list[int]:
    """
    n_digit桁のランダム整数をsample_rateの確率で生成する
    メモリ負荷も考えて、10000サンプル以上は生成しないようにする。
    """
    unique_numbers = set()

    start = 10 ** (n_digit - 1)
    stop = 10**n_digit - 1
    print(f"start: {start}, stop: {stop}, sample_rate: {sample_rate}")
    sample_num = int((stop - start) * sample_rate)
    if sample_num > 10000:
        raise ValueError(f"sample_numが大きすぎます⇒: {sample_num}")

    while len(unique_numbers) < sample_num:
        num = random.randint(start, stop)
        unique_numbers.add(num)

    return list(unique_numbers)


@dataclass
class 足し算生成桁数と確率:
    digit1: int
    digit2: int
    sample_rate1: float
    sample_rate2: float


def 二項n桁の足し算の文字列生成(data: 足し算生成桁数と確率) -> str:
    digit1 = data.digit1
    digit2 = data.digit2
    sample_rate1 = data.sample_rate1
    sample_rate2 = data.sample_rate2

    digit1_sampled = n桁のランダム整数生成(digit1, sample_rate1)
    digit2_sampled = n桁のランダム整数生成(digit2, sample_rate2)

    sample_num = len(digit1_sampled) * len(digit2_sampled)
    print(f"{digit1}桁+{digit2}桁のサンプル数: {sample_num}")
    if sample_num > 10000:
        raise ValueError("サンプル数が多すぎる")

    rslt = ""
    for i in digit1_sampled:
        for j in digit2_sampled:
            ques_part = f"{i}+{j}="
            ans_part = f"{bs_tok}{i+j}{es_tok}"
            rslt += f"{ques_part}{ans_part}\n"

            # 交換法則を学習させるため、入れ替えた値も書き込み
            ques_part = f"{j}+{i}="
            ans_part = f"{bs_tok}{j+i}{es_tok}"
            rslt += f"{ques_part}{ans_part}\n"
    return rslt


class 足し算データ生成:
    def __init__(self, data_path: Path):
        self.data_path = data_path

    def any_digitwrite_files(self, gen_parm: 足し算生成桁数と確率):
        adding_strings = 二項n桁の足し算の文字列生成(gen_parm)
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

        with open(self.data_path / f"{gen_parm.digit1}桁{gen_parm.digit2}桁.txt", "w") as f:
            f.write(adding_strings)

    def any_term_write_files(self, term_num: int, sample_rate: float):
        adding_strings = 多項の足し算の文字列生成(term_num, sample_rate)
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

        with open(self.data_path / f"{term_num}項.txt", "w") as f:
            f.write(adding_strings)


def 桁数の内挿(dir_path: str):
    """
    桁数の内挿汎化について
    1,2,4,6桁の足し算を学習させて3,5,7桁の足し算が出来るか検証する
    1桁 - 9 * 9= 約100サンプルある。原子的な演算なのですべてのサンプルをデータセットに含める。
    1桁+2桁 or 2桁+1桁 - 9 * 99 * 2=約2000サンプルある。原子的な演算なのですべてのサンプルをデータセットに含める。
    2桁-99*99=約10000サンプル。30%をデータセットに含める。
    """

    data_path = Path(dir_path) / "桁数の内挿"
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    train_path = data_path / "train"
    if train_path.exists():
        shutil.rmtree(train_path)

    test_path = data_path / "test"
    if test_path.exists():
        shutil.rmtree(test_path)

    # train_write_path = functools.partial(write_file, train_path)
    train_data_gen = 足し算データ生成(train_path)

    gen_param = [
        [1, 1, 1.0, 1.0],
        [2, 1, 1.0, 1.0],
        [2, 2, 0.3, 0.3],
        [4, 1, 0.01, 1.0],
        [4, 2, 0.01, 0.1],
        [4, 4, 3e-3, 3e-3],
        [6, 1, 1e-4, 1.0],
        [6, 2, 1e-4, 0.1],
        [6, 4, 1e-4, 1e-3],
        [6, 6, 3e-5, 3e-5],
    ]
    for param in gen_param:
        train_data_gen.any_digitwrite_files(足し算生成桁数と確率(*param))

    gen_param = [
        [3, 1, 0.1, 1.0],
        [3, 2, 0.1, 0.1],
        [3, 3, 0.01, 0.01],
        [5, 1, 1e-4, 1.0],
        [5, 2, 1e-4, 0.1],
        [5, 3, 1e-4, 0.01],
        [5, 4, 1e-4, 1e-3],
        [5, 5, 1e-4, 1e-4],
    ]

    test_data_gen = 足し算データ生成(test_path)
    for param in gen_param:
        test_data_gen.any_digitwrite_files(足し算生成桁数と確率(*param))


def テスト_桁数の内挿():
    桁数の内挿("dataset/")


def 桁数の外挿汎化(dir_path: str):
    """
    桁数の外挿汎化について
    1桁、2桁,3桁,4桁の足し算を学習させて5桁の足し算ができるのか
    """

    data_path = Path(dir_path) / "桁数の外挿汎化"
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    train_path = data_path / "train"
    if train_path.exists():
        shutil.rmtree(train_path)

    test_path = data_path / "test"
    if test_path.exists():
        shutil.rmtree(test_path)

    gen_param = [
        [1, 1, 1.0, 1.0],
        [2, 1, 1.0, 1.0],
        [2, 2, 0.3, 0.3],
        [3, 1, 0.1, 1.0],
        [3, 2, 0.1, 0.1],
        [3, 3, 0.01, 0.01],
        [4, 1, 0.01, 1.0],
        [4, 2, 0.01, 0.1],
        [4, 3, 3e-3, 0.01],
        [4, 4, 3e-3, 3e-3],
    ]
    train_data_gen = 足し算データ生成(train_path)
    for param in gen_param:
        train_data_gen.any_digitwrite_files(足し算生成桁数と確率(*param))

    gen_param = [
        [5, 1, 1e-4, 1.0],
        [5, 2, 1e-4, 0.1],
        [5, 3, 1e-4, 0.01],
        [5, 4, 1e-4, 1e-3],
        [5, 5, 1e-4, 1e-4],
    ]
    test_data_gen = 足し算データ生成(test_path)
    for param in gen_param:
        test_data_gen.any_digitwrite_files(足し算生成桁数と確率(*param))


def 多項の足し算の文字列生成(term_num: int, sample_rate: float) -> str:
    if term_num < 2:
        raise ValueError("項数は2以上である必要がある")

    randints = n桁のランダム整数生成(term_num, sample_rate)

    rslt = ""
    for one_randint in randints:
        ans = 0
        one_randint_str = str(one_randint)
        for i in one_randint_str:
            ans += int(i)
            rslt += f"{i}+"

        rslt += f"={bs_tok}{ans}{es_tok}\n"

    return rslt


def 項数の内挿汎化(dir_path: str):
    data_path = Path(dir_path) / "項数の内挿汎化"
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    train_path = data_path / "train"
    if train_path.exists():
        shutil.rmtree(train_path)
    test_path = data_path / "test"
    if test_path.exists():
        shutil.rmtree(test_path)

    train_gen_param = [
        [1, 1.0],
        [2, 1.0],
        [4, 0.1],
        [6, 1e-3],
        [8, 1e-5],
    ]
    train_generator = 足し算データ生成(train_path)
    for param in train_gen_param:
        train_generator.any_term_write_files(param[0], param[1])

    train_gen_param2 = [
        [2, 1, 1.0, 1.0],
        [2, 2, 0.3, 0.3],
    ]
    for param in train_gen_param2:
        train_generator.any_digitwrite_files(足し算生成桁数と確率(*param))

    test_gen_param = [
        [3, 0.1],
        [5, 1e-4],
        [7, 1e-6],
    ]
    test_generator = 足し算データ生成(test_path)
    for param in test_gen_param:
        test_generator.any_term_write_files(param[0], param[1])


def 項数の外挿汎化(dir_path: str):
    """
    項数の外挿汎化
    0~9までの数の足し算を5項まで学習する。
    それとは別に２桁までの足し算は学習しておく。1~9までの数が11項の足し算は最大9×11で99になるのでその演算が出来るようにしておく。
    最後に10項の足し算が出来るのか検証する。
    0~9までの数の足し算がn項ある場合、サンプル数は10^nとなる。
    学習したかの判定について
    対象の演算のなかで
    • 100サンプル
    • 学習に使用していないサンプルで対象の演算サンプル全体の1%
    の数の大きい方を用いて、正答率を検証する。
    正答率が90%以上ならその演算は出来ると見なす。
    """
    data_path = Path(dir_path) / "項数の外挿汎化"
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)

    train_path = data_path / "train"
    if train_path.exists():
        shutil.rmtree(train_path)
    test_path = data_path / "test"
    if test_path.exists():
        shutil.rmtree(test_path)

    train_gen_param = [
        [1, 1.0],
        [2, 1.0],
        [3, 1.0],
        [4, 0.1],
        [5, 1e-2],
    ]
    train_generator = 足し算データ生成(train_path)
    for param in train_gen_param:
        train_generator.any_term_write_files(param[0], param[1])

    test_gen_param = [
        [6, 1e-3],
        [7, 1e-4],
    ]
    test_generator = 足し算データ生成(test_path)
    for param in test_gen_param:
        test_generator.any_term_write_files(param[0], param[1])


def 足し算ドリルを生成(dir_path: str, limit_num: int):
    with open(dir_path + "足し算ドリル.txt", "w") as f:
        for i in range(limit_num):
            for j in range(limit_num):
                ques_part = f"{i}+{j}="
                ans_part = f"{bs_tok}{i+j}{es_tok}"
                f.write(f"{ques_part}{ans_part}\n")


def テスト_足し算ドリルを生成():
    足し算ドリルを生成("dataset/", 10)


def make_causal_text(text: str) -> list[str]:
    """
    与えられたtextに対して、bs_tokより後のtextが一文字ずつ増やしたtextのlistを返す
    例
    causal_text = make_causal_text("1+2=<3>")
    assert causal_text[1] == "1+2=<3"
    assert causal_text[2] == "1+2=<3>"
    assert len(causal_text) == 2

    Args:
        text (str): text

    Returns:
        list[str]: textを一文字ずつ増やしたtextのlist
    """
    bs_tok_pos = text.index(bs_tok) + 1
    es_tok_pos = text.index(es_tok) + 1
    q_part = text[:bs_tok_pos]  # question part
    a_part = text[bs_tok_pos:es_tok_pos]  # answer part
    ans_len = len(a_part)

    text_list = []
    for i in range(ans_len):
        ans = a_part[: i + 1]
        ques_and_masked_ans = q_part + ans
        text_list.append(ques_and_masked_ans)

    return text_list


def テスト_make_causal_text():
    causal_text = make_causal_text("1+2=<3>")
    assert causal_text[0] == "1+2=<3"
    assert causal_text[1] == "1+2=<3>"
    assert len(causal_text) == 2


class 足し算ドリル(Dataset):
    def __init__(self, seq_len: int):
        original_text = Path("dataset/足し算ドリル.txt").read_text()
        text_list: list[str] = original_text.split("\n")
        dataset = []
        for text in text_list:
            if text == "":
                continue
            dataset.extend(make_causal_text(text))

        self.dataset = dataset
        self.seq_len = seq_len

        self.tokenizer = Tokenizer(original_text)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        data = self.dataset[idx]
        encoded = self.tokenizer.encode(data)
        causal_data = self.tokenizer.make_causal_data(encoded, self.seq_len)

        return causal_data


def テスト_足し算ドリル():
    from pprint import pprint
    dataset = 足し算ドリル(15)
    for data in dataset:
        pprint(data)
        print(dataset.tokenizer.decode(data["token_ids"]))
        break
