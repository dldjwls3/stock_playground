from pytorch_lightning import Trainer, seed_everything
from model.baseline import Baseline
from pytorch_lightning.logging import CometLogger
from dataset import KOSPI200Dataset

# @hydra.main(config_path='./hydra/config.yaml')
# def parse_config(cfg: DictConfig):
#     print(cfg.pretty())
#
seed = 11
train_length = 300
val_length = 30
test_length = 585
sequence_length = 9

hidden_layer = 6
hidden_feature = 12
activation = 'tanh'

if __name__ == '__main__':
    seed_everything(seed)
    KOSPI200Dataset.setup(
        train_length=train_length,
        val_length=val_length,
        test_length=test_length,
        sequence_length=sequence_length
    )  # 2020.07.03 615일(개장일 기준) 이전이 2018.01.02

    comet_logger = CometLogger(
        api_key="SWhvV0XPkHV8tPdU8Nv67EXxU",
        workspace="dldjwls3",  # Optional
        project_name="stock-test",
        experiment_name='test'
    )

    model = Baseline(
        seed=seed,
        sequence_length=sequence_length,
        num_feature=5,
        activation=activation,
        hidden_layer=hidden_layer,
        hidden_feature=hidden_feature
    )
    trainer = Trainer(max_epochs=120, gpus=-1, logger=comet_logger)
    trainer.fit(model)

    # trainer.test(model)