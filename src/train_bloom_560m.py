# %%

from util.utils import ini_setting
from bloom_560m import Bloom560m
from bloom560m_easyEN2SP import get_trainer, MLFlowExperimentManager
from easy_ds_EN_to_SP import EasyEnToSpDM
from data_process.arc.arc_preprocess import ArcTaskDataModule



def run_training():
    model = Bloom560m("D:/models", lr=0.001)
    data_module = EasyEnToSpDM(tokenizer=model.tokenizer, batch_size=1)
    test_data_module = ArcTaskDataModule(data_path="dataset/Selective-Arc/original_arc/training", batch_size=1)
    trainer = get_trainer()

    with MLFlowExperimentManager():
        trainer.fit(model=model, datamodule=data_module)
        trainer.test(model=model, datamodule=test_data_module)
        #mlflow.pytorch.autologでlogをしているが,その説明にはfitの内容だけlogすると書いている.ただ実際はtestでもlogがされている.この部分はarcのtestすると良いかな?

if __name__ == "__main__":
    # ini_setting()
    # run_training()
    # pass

    # test_data_module = ArcTaskDataModule(data_path="dataset/Selective-Arc/original_arc/training", batch_size=1)
    # test_data_module.setup()
    # print(test_data_module.test_dataloader())

    model = Bloom560m("D:/models", lr=0.001)
    EasyEnToSpDM()

