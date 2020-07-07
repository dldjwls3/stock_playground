from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from baseline import Baseline
from comet_ml import Optimizer
from pytorch_lightning.logging import CometLogger
from dataset import KOSPI200Dataset
import hydra
from omegaconf import DictConfig

seed_everything(1)

# @hydra.main(config_path='./hydra/config.yaml')
# def parse_config(cfg: DictConfig):
#     print(cfg.pretty())
#

config = {
    'algorithm': 'bayes',
    'parameters': {
        'hidden_layer': {'type': 'integer', 'min': 1, 'max': 100},
        'hidden_feature': {'type': 'integer', 'min': 1, 'max': 100},
        'activation': {'type': 'categorical', 'values': ['sigmoid', 'tanh']}
    },
    'spec': {
        'seed': 1,
        'metric': 'val_loss',
        'objective': 'minimize'
    }
}

train_days=1000

if __name__ == '__main__':

    opt = Optimizer(config, api_key='SWhvV0XPkHV8tPdU8Nv67EXxU', project_name=f'stock-gcn-experiment-{train_days}days')
    for experiment in opt.get_experiments():
        print('start')
        KOSPI200Dataset.set_data_split(train_days, 30, 585)  # 2020.07.03 615일(개장일 기준) 이전이 2018.01.02

        hidden_layer = experiment.get_parameter('hidden_layer')
        hidden_feature = experiment.get_parameter('hidden_feature')
        activation = experiment.get_parameter('activation')

        comet_logger = CometLogger(
            api_key="SWhvV0XPkHV8tPdU8Nv67EXxU",
            workspace="dldjwls3",  # Optional
            project_name=f"stock-gcn-experiment-{train_days}days",
            experiment_name=f'gcn_{activation}_{hidden_layer}_{hidden_feature}',
            experiment_key=experiment.get_key()
        )

        model = Baseline(hidden_layer=hidden_layer, hidden_feature=hidden_feature, activation=activation)
        trainer = Trainer(max_epochs=120, gpus=-1, logger=comet_logger)
        trainer.fit(model)



    # trainer.test(model)