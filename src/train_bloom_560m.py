# %%

from util.utils import ini_setting
from bloom_560m import Bloom560m
from bloom560m_easyEN2SP import get_trainer, MLFlowExperimentManager
from easy_ds_EN_to_SP import EasyEnToSpDM



def run_training():
    model = Bloom560m("D:/models", lr=0.001)
    data_module = EasyEnToSpDM(tokenizer=model.tokenizer, batch_size=1)
    trainer = get_trainer()

    with MLFlowExperimentManager():
        trainer.fit(model=model, datamodule=data_module)
        trainer.test(model=model, datamodule=data_module)
        #mlflow.pytorch.autologでlogをしているが,その説明にはfitの内容だけlogすると書いている.ただ実際はtestでもlogがされている.この部分はarcのtestすると良いかな?
    # predictions = trainer.predict(model=model, datamodule=data_module, return_predictions=True)
        # mlflow.log_param("aiueo", 1)
        
        

    # display_predictions(predictions)

def display_predictions(predictions):
    for prediction in predictions:
        print(prediction)

if __name__ == "__main__":
    ini_setting()
    run_training()
    pass
