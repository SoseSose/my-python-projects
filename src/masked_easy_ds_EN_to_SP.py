import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from lightning import LightningDataModule


def tokenizer_settings(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "mask_token": "<mask>",
            "pad_token": "<pad>",
        }
    )
    return tokenizer


def wrap_with_special_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    return tokenizer.bos_token + text + tokenizer.eos_token


def mask_data_collater(
    text_ids: list[int],
    # text状態で入れ替えるのはtokenずつ変更にはならない可能性が高いから、text_ids状態で受け取る必要がある。
    tokenizer: PreTrainedTokenizerBase,
    prob: float = 0.15,
    not_learn_id: int = -100,
) -> tuple[list[int], list[int]]:
    """
    トークナイズされた、テキストのリストを受け取り、bos_tokenからeos_tokenまでの間を指定された確率でトークンをマスクします。

    Args:
        text_ids (list[int]): トークン化されたテキストのIDリスト。
        tokenizer (PreTrainedTokenizerBase): トークナイザーオブジェクト。
        prob (float, optional): トークンをマスクする確率。デフォルトは0.15。
        not_learn_id (int, optional): マスクされていないトークンのラベル。デフォルトは-100。

    Raises:
        ValueError: tokenizerにbos_token_id, eos_token_id, またはmask_token_idが設定されていない場合。
        ValueError: probが0以上1以下でない場合。

    Returns:
        tuple[list[int], list[int]]: マスクされたテキストIDのリストと、対応するラベルのリスト。
    """
    # text状態で入れ替えるのはtokenずつ変更にはならない可能性が高いから、text_ids状態で受け取る必要がある。
    if tokenizer.bos_token_id is None:
        raise ValueError("tokenizerにbos_token_idが設定されていません。")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizerにeos_token_idが設定されていません。")
    if tokenizer.mask_token_id is None:
        raise ValueError("tokenizerにmask_token_idが設定されていません。")

    if not (0 <= prob <= 1):
        raise ValueError("probは0以上1以下でなければなりません。")

    masked_text = []
    labels = []

    in_sequence = False

    for token in text_ids:
        if token == tokenizer.eos_token_id:
            in_sequence = False

        if in_sequence and random.random() < prob:
            masked_text.append(tokenizer.mask_token_id)
            labels.append(token)
        else:
            masked_text.append(token)
            labels.append(not_learn_id)

        # もしbos_token_idがある場合は、そのあとからはin_sequence=Trueとする
        if token == tokenizer.bos_token_id:
            in_sequence = True

    return masked_text, labels


class TranslationDataset(Dataset):
    ENGLISH_WORDS = ["dog", "water", "mother", "hello", "tree"]
    SPANISH_WORDS = ["perro", "agua", "madre", "hola", "árbol"]

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer_settings(tokenizer)

    def __len__(self):
        return len(self.ENGLISH_WORDS)

    def __getitem__(self, idx):
        eng = self.ENGLISH_WORDS[idx]
        spa = self.SPANISH_WORDS[idx]
        text = f'How do you say "{eng}" in Spanish?\n\n'
        text += wrap_with_special_tokens(f"{spa}", self.tokenizer)
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=16,  # batch中でサイズをそろえるようにとりあえずmax_lengthを指定,batch_encodeを使ったほうがよさそうではあるけど動作しているうちは直さない、https://huggingface.co/docs/transformers/pad_truncation
            return_attention_mask=True,
            # return_tensors="np"
        )

        masked_text, labels = mask_data_collater(
            list(encoded_text["input_ids"]), self.tokenizer, prob=1.0
        )

        return {
            "input_ids": torch.tensor(masked_text),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(encoded_text["attention_mask"]),
        }


def get_translation_dataloader(
    tokenizer: PreTrainedTokenizerBase, batch_size=1, shuffle=False
):
    dataset = TranslationDataset(tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    return dataloader


class MaskedEasyEnToSpDM(LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.dl = get_translation_dataloader(
            self.tokenizer,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def train_dataloader(self):
        return self.dl

    def val_dataloader(self):
        return self.dl

    def test_dataloader(self):
        return self.dl

    def predict_dataloader(self):
        return self.dl


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    dataloader = get_translation_dataloader(tokenizer, batch_size=2)
    for data in dataloader:
        # print("a")
        # print(data)
        print(data["input_ids"].shape)
        print(data["labels"].shape)
        print(data["attention_mask"].shape)
        print(tokenizer.batch_decode(data["input_ids"]))
        print(tokenizer.batch_decode(data["labels"]))
