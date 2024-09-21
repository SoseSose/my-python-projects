import shutil
from pathlib import Path

from bloom560m_easyEN2SP import MLFlowExperimentManager, get_trainer
from bloom_560m import Bloom560m
from data_process.arc.arc_preprocess import ArcTaskDataModule, ArcTaskDataset
from masked_easy_ds_EN_to_SP import MaskedEasyEnToSpDM
from util.utils import ini_setting


def run_training():
    model = Bloom560m("C:/Users/音声入力用アカウント/Documents/models", lr=0.0001)

    train_data_module = MaskedEasyEnToSpDM(tokenizer=model.tokenizer, batch_size=5)
    # test_data_module = MaskedEasyEnToSpDM(tokenizer=model.tokenizer, batch_size=1)
    test_data_module = ArcTaskDataModule(
        tokenizer=model.tokenizer,
        data_path="dataset/Selective-Arc/original_arc/training",
        batch_size=16,
    )
    trainer = get_trainer()
    print("start training")

    trainer.fit(model=model, datamodule=train_data_module)
    trainer.test(model=model, datamodule=test_data_module)

    # 勝手にmlrunsディレクトリが作成されて邪魔なため削除する
    mlruns_dir = Path("mlruns")
    if mlruns_dir.exists() and mlruns_dir.is_dir():
        shutil.rmtree(mlruns_dir)

    # model.model.to("cuda")
    # ENGLISH_WORDS = ["dog", "water", "mother", "hello", "tree"]
    # SPANISH_WORDS = ["perro", "agua", "madre", "hola", "árbol"]
    # for i,data in enumerate(zip(ENGLISH_WORDS, SPANISH_WORDS)):
    #     eng, spa = data
    #     text = f'How do you say "{eng}" in Spanish?\n<s>'
    #     data = model.tokenizer.encode_plus(text, return_tensors="pt")
    #     input = data["input_ids"].to("cuda")
    #     print(f"input: {i}")
    #     print("question:")
    #     print(model.tokenizer.decode(input[0]))
    #     text = model.model.generate(input, max_length=100)
    #     print("answer:")
    #     print(model.tokenizer.decode(text[0]))


if __name__ == "__main__":
    ini_setting()
    run_training()
    # pass
