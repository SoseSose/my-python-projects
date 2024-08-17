# %%
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import  PreTrainedTokenizerBase
from lightning import LightningDataModule

INSTRUCTION_TEMPLATE_BASE = "\n\n### Human:"
RESPONSE_TEMPLATE_BASE = "\n\n### Assistant:"
ENGLISH_WORDS = ["dog", "water", "mother", "hello", "tree"]
SPANISH_WORDS = ["perro", "agua", "madre", "hola", "árbol"]


def easy_ds():
    origin_str = [
        f'\n\n### Human: How do you say "{Eng}" in Spanish?\n\n### Assistant: {Spa}'
        for Eng, Spa in zip(ENGLISH_WORDS, SPANISH_WORDS)
    ]

    train_data = {
        "text": origin_str,
    }
    return Dataset.from_dict(train_data)


def add_special_tokens(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    # add eos_token before human text and bos_token before assistant text
    example["text"] = (
        example["text"]
        .replace(
            INSTRUCTION_TEMPLATE_BASE, tokenizer.eos_token + INSTRUCTION_TEMPLATE_BASE
        )
        .replace(RESPONSE_TEMPLATE_BASE, RESPONSE_TEMPLATE_BASE + tokenizer.bos_token)
    )
    if not example["text"].endswith(tokenizer.eos_token):
        example["text"] += tokenizer.eos_token
    # Remove leading EOS tokens
    while example["text"].startswith(tokenizer.eos_token):
        example["text"] = example["text"][len(tokenizer.eos_token) :]

    return example


def add_special_tokens_to_ds(ds, tokenizer):
    return ds.map(lambda x: add_special_tokens(x, tokenizer))


def tokenized_ds(ds, tokenizer):
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
    ) # convert dataset from lists to torch tensors
    return ds


def get_masked_ds(tokenizer):
    ds = easy_ds()
    ds = add_special_tokens_to_ds(ds, tokenizer)
    ds = tokenized_ds(ds, tokenizer)
    ds = add_labels_to_ds(ds)
    ds = masked_ds(ds, tokenizer)
    return ds


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
