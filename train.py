from pytorch_lightning import Trainer, seed_everything
from model.baseline import Baseline
from pytorch_lightning.logging import CometLogger
from dataset import KOSPI200Dataset

seed_everything(1)

# @hydra.main(config_path='./hydra/config.yaml')
# def parse_config(cfg: DictConfig):
#     print(cfg.pretty())
#
train_length = 500
val_length = 30
test_length = 585
sequence_length = 5

if __name__ == '__main__':

    KOSPI200Dataset.setup(
        train_length=train_length,
        val_length=val_length,
        test_length=test_length,
        sequence_length=5
    )  # 2020.07.03 615일(개장일 기준) 이전이 2018.01.02

    comet_logger = CometLogger(
        api_key="SWhvV0XPkHV8tPdU8Nv67EXxU",
        workspace="dldjwls3",  # Optional
        project_name="stock-test",
        experiment_name='test'
    )

    model = Baseline(sequence_length=sequence_length, num_feature=5, activation='sigmoid', hidden_layer=30, hidden_feature=39)
    trainer = Trainer(max_epochs=120, gpus=-1, logger=comet_logger)
    trainer.fit(model)

    # trainer.test(model)