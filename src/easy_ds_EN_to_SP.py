# %%
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from lightning import LightningDataModule

INSTRUCTION_TEMPLATE_BASE = "\n\n### Human:"
RESPONSE_TEMPLATE_BASE = "\n\n### Assistant:"
ENGLISH_WORDS = ["dog", "water", "mother", "hello", "tree"]
SPANISH_WORDS = ["perro", "agua", "madre", "hola", "árbol"]


def easy_ds():
    origin_str = [
        f'{INSTRUCTION_TEMPLATE_BASE} How do you say "{Eng}" in Spanish?\n\n{RESPONSE_TEMPLATE_BASE} {Spa}'
        for Eng, Spa in zip(ENGLISH_WORDS, SPANISH_WORDS)
    ]

    train_data = {
        "text": origin_str,
    }
    return Dataset.from_dict(train_data)


def test_easy_ds():
    ds = easy_ds()
    for i in ds:
        assert (
            i["text"]
            == '\n\n### Human: How do you say "dog" in Spanish?\n\n### Assistant: perro'
        )
        break


def add_special_tokens(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    text = example["text"]

    text = text.replace(
        INSTRUCTION_TEMPLATE_BASE, tokenizer.eos_token + INSTRUCTION_TEMPLATE_BASE
    )
    text = text.replace(
        RESPONSE_TEMPLATE_BASE, RESPONSE_TEMPLATE_BASE + tokenizer.bos_token
    )

    if not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    # Remove leading EOS tokens
    while text.startswith(tokenizer.eos_token):
        text = text[len(tokenizer.eos_token) :]

    return {"text": text}


def add_special_tokens_to_ds(ds, tokenizer):
    return ds.map(lambda x: add_special_tokens(x, tokenizer))


def tokenized_ds(ds, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return ds.map(
        lambda example: tokenizer(example["text"], padding=True),
        batched=True,
        remove_columns=["text"],
        #! padding=Trueを追加している.
    )


def add_labels_to_ds(ds):
    return ds.map(lambda x: {"labels": x["input_ids"]}, batched=True)


def create_special_mask(tokenizer, example: dict) -> dict:
    """Mask human text and keep assistant text as it is.

    Args:
        example (Dict): Result of tokenizing some text

    Returns:
        Dict: The dict with the label masked
    """
    # setting a token to -100 is how we "mask" a token
    # and tell the model to ignore it when calculating the loss
    mask_token_id = -100
    # assume we always start with a human text
    human_text = True
    for idx, tok_id in enumerate(example["labels"]):
        if human_text:
            # mask all human text up until and including the bos token
            example["labels"][idx] = mask_token_id
            if tok_id == tokenizer.bos_token_id:
                human_text = False
        elif not human_text and tok_id == tokenizer.eos_token_id:
            human_text = True
        elif not human_text:
            # leave example['labels'] text as it is when assistant text
            continue
    return example


def masked_ds(ds, tokenizer):
    ds = ds.map(lambda x: create_special_mask(tokenizer, x))
    ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )  # convert dataset from lists to torch tensors
    return ds


def get_masked_ds(tokenizer):
    ds = easy_ds()
    ds = add_special_tokens_to_ds(ds, tokenizer)
    ds = tokenized_ds(ds, tokenizer)
    ds = add_labels_to_ds(ds)
    ds = masked_ds(ds, tokenizer)
    return ds


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    print(tokenizer.special_tokens_map)
    tokenizer.add_special_tokens({"mask_token": "<mask>"})
    print(tokenizer.mask_token)
    print(tokenizer.mask_token_id)
    # %%
    ds = get_masked_ds(tokenizer)
    # tokenizer.pad_token = "<pad>"
    # tokenizer.pad_token_id = -100
    for i in ds:
        print(i["input_ids"])
        print(tokenizer.decode(i["input_ids"]))
        print(i["labels"])
        print(i["attention_mask"])
        # print(tokenizer.decode(i["labels"]))
        # print(tokenizer.decode(i["attention_mask"]))
# %%


class EasyEnToSpDM(LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, batch_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def prepare_data(self):
        self.ds = get_masked_ds(self.tokenizer)

    def setup(self, stage: str):
        self.dl = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=4,#このdsでは不要
        )

    def train_dataloader(self):
        return self.dl

    def val_dataloader(self):
        return self.dl

    def test_dataloader(self):
        return self.dl

    def predict_dataloader(self):
        return self.dl


import random


def mask_tokens(
    text: list[int],
    tokenizer: PreTrainedTokenizerBase,
    prob: float = 0.15,
    not_learn_id: int = -100,
) -> tuple[list[int], list[int]]:
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

    for token in text:
        if token == tokenizer.bos_token_id:
            in_sequence = True

        if in_sequence and random.random() < prob:
            masked_text.append(tokenizer.mask_token_id)
            labels.append(token)
        else:
            masked_text.append(token)
            labels.append(not_learn_id)

        if token == tokenizer.eos_token_id:
            in_sequence = False

    return masked_text, labels
