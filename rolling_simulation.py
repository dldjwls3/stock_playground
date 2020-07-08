from pytorch_lightning import Trainer, seed_everything
from model.baseline import Baseline
from pytorch_lightning.logging import CometLogger
from dataset import KOSPI200Dataset
import hydra
from omegaconf import DictConfig


@hydra.main()
def main(cfg: DictConfig):
    for i in range(20):
        train_length = 300
        val_length = 30
        test_length = 585 - 30 * i

        seed = 1017
        seed_everything(seed)
        sequence_length = 10
        hidden_layer = 5
        hidden_feature = 12
        activation = 'sigmoid'

        KOSPI200Dataset.setup(
            train_length=train_length,
            val_length=val_length,
            test_length=test_length,
            sequence_length=sequence_length
        )  # 2020.07.03 615일(개장일 기준) 이전이 2018.01.02

        comet_logger = CometLogger(
            api_key="SWhvV0XPkHV8tPdU8Nv67EXxU",
            workspace="dldjwls3",  # Optional
            project_name=f"stock-rolling-simulation-{train_length}days",
            experiment_name=f'{seed}seed_gcn_{sequence_length}_{activation}_{hidden_layer}_{hidden_feature}'
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
        trainer.test(model)


if __name__ == '__main__':
    main()
