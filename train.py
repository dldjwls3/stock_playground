from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from baseline import Baseline
from comet_ml import Experiment
from pytorch_lightning.logging import CometLogger
from dataset import KOSPI200Dataset

seed_everything(1)
KOSPI200Dataset.set_data_split(500, 30, 30)

if __name__ == '__main__':
    comet_logger = CometLogger(
        api_key="SWhvV0XPkHV8tPdU8Nv67EXxU",
        workspace="dldjwls3",  # Optional
        project_name="stock_playground",  # Optional
        # rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        experiment_name="default" # Optional
    )

    model = Baseline()
    trainer = Trainer(max_epochs=120, gpus=-1, logger=comet_logger)
    trainer.fit(model)

    # trainer.test(model)