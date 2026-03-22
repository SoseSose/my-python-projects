from pathlib import Path

# import torch
from torch import Tensor as torch_Tensor
from torch import bool as torch_bool
from torch import full as torch_full
from torch import long as torch_long
from torch import tensor as torch_tensor
from torch import zeros as torch_zeros
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

    def encode(self, val: str) -> torch_Tensor:
        ints = [self.str_to_int[c] for c in val]
        return torch_tensor(ints, dtype=torch_long)

    def make_causal_data(self, token_ids: torch_Tensor, seq_len: int) -> dict:
        original_len = len(token_ids)

        # 最後のtokenを除いてseq_lenまでpadする
        x = token_ids[:-1]
        pad_id = self.str_to_int[self.pad_token]
        x = F.pad(x, (0, seq_len - original_len + 1), "constant", pad_id)

        mask = torch_zeros(seq_len, dtype=torch_bool)
        mask[: original_len - 1] = 1

        target = torch_full((seq_len,), -100, dtype=torch_long)
        target[original_len - 1] = token_ids[-1]

        return {"token_ids": x, "mask": mask, "targets": target}

    def decode(self, val: torch_Tensor) -> str:
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
