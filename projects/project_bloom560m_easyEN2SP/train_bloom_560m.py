from util.utils import ini_setting
from bloom_560m import Bloom560m
from bloom560m_easyEN2SP import get_trainer, MLFlowExperimentManager
from masked_easy_ds_EN_to_SP import MaskedEasyEnToSpDM
from data_process.arc.arc_preprocess import ArcTaskDataModule
from pathlib import Path


def run_training():
    
    model = Bloom560m("C:/Users/音声入力用アカウント/Documents/models", lr=0.001)
    train_data_module = MaskedEasyEnToSpDM(tokenizer=model.tokenizer, batch_size=1)

    test_data_module = ArcTaskDataModule(tokenizer=model.tokenizer, data_path="dataset/Selective-Arc/original_arc/training", batch_size=1)
    # test_data_module = MaskedEasyEnToSpDM(tokenizer=model.tokenizer, batch_size=1)
    ckpt_path = "logs/checkpoints"
    trainer, check_point_id = get_trainer(ckpt_path=ckpt_path)
    
    print("start training")


    with MLFlowExperimentManager():
        
        trainer.fit(model=model, datamodule=train_data_module)
        # trainer.test(model=model, datamodule=test_data_module)

        for ckpt_file_name in Path(ckpt_path).glob(f"{check_point_id}*.ckpt"):
            #学習後期は逆に過学習していたりするので、最良だったモデルでtestする。
            best_model = Bloom560m.load_from_checkpoint(ckpt_file_name)
            trainer.test(model=best_model, datamodule=test_data_module)
        
        #mlflow.pytorch.autologでlogをしているが,その説明にはfitの内容だけlogすると書いている.ただ実際はfitした後にtestすれば,testでもlogがされている.
        #システムメトリクスも自動で取れるのでこちらを使用する.


    best_model.model.to("cuda")
    ENGLISH_WORDS = ["dog", "water", "mother", "hello", "tree"]
    SPANISH_WORDS = ["perro", "agua", "madre", "hola", "árbol"]

    for i,data in enumerate(zip(ENGLISH_WORDS, SPANISH_WORDS)):
        eng, spa = data
        text = f'How do you say "{eng}" in Spanish?\n\n<s>'
        data = best_model.tokenizer.encode_plus(text, return_tensors="pt")
        input = data["input_ids"].to("cuda")
        print(f"input: {i}")
        print("question:")
        print(best_model.tokenizer.decode(input[0]))
        text = best_model.model.generate(input, max_length=100)
        print("answer:")
        ans_text = best_model.tokenizer.decode(text[0])
        ans_text = ans_text[ans_text.find("\n\n<s>"):]
        print(ans_text)

if __name__ == "__main__":
    ini_setting()
    run_training()
    # pass
