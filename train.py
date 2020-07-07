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

if __name__ == '__main__':

    KOSPI200Dataset.set_data_split(500, 30, 585)  # 2020.07.03 615일(개장일 기준) 이전이 2018.01.02

    comet_logger = CometLogger(
        api_key="SWhvV0XPkHV8tPdU8Nv67EXxU",
        workspace="dldjwls3",  # Optional
        project_name="stock-gcn-experiment",
        experiment_name='test'
    )

    model = Baseline()
    trainer = Trainer(max_epochs=120, gpus=-1, logger=comet_logger)
    trainer.fit(model)

    # trainer.test(model)