from pytorch_lightning import Trainer, seed_everything
from model.baseline import Baseline
from comet_ml import Optimizer
from pytorch_lightning.logging import CometLogger
from dataset import KOSPI200Dataset
import hydra
from omegaconf import DictConfig

config = {
    'algorithm': 'bayes',
    'parameters': {
        'hidden_layer': {'type': 'integer', 'min': 1, 'max': 10},
        'hidden_feature': {'type': 'integer', 'min': 1, 'max': 20},
        'activation': {'type': 'categorical', 'values': ['sigmoid', 'tanh']},
        'sequence_length': {'type': 'integer', 'min': 2, 'max': 10}
    },
    'spec': {
        'seed': 1,
        'metric': 'val_loss',
        'objective': 'minimize',
        'retryLimit': 100
    }
}

@hydra.main()
def main(cfg: DictConfig):
    train_length = 500
    val_length = 30
    test_length = 585

    opt = Optimizer(
        config,
        api_key='SWhvV0XPkHV8tPdU8Nv67EXxU',
        project_name=f'stock-gcn-experiment-sequences-{train_length}days'
    )
    for experiment in opt.get_experiments():
        seed = 1
        seed_everything(seed)

        hidden_layer = experiment.get_parameter('hidden_layer')
        hidden_feature = experiment.get_parameter('hidden_feature')
        activation = experiment.get_parameter('activation')
        sequence_length = experiment.get_parameter('sequence_length')

        KOSPI200Dataset.setup(
            train_length=train_length,
            val_length=val_length,
            test_length=test_length,
            sequence_length=sequence_length
        )  # 2020.07.03 615일(개장일 기준) 이전이 2018.01.02

        comet_logger = CometLogger(
            api_key="SWhvV0XPkHV8tPdU8Nv67EXxU",
            workspace="dldjwls3",  # Optional
            project_name=f'stock-gcn-experiment-sequences-{train_length}days',
            experiment_name=f'gcn_{sequence_length}_{activation}_{hidden_layer}_{hidden_feature}',
            experiment_key=experiment.get_key()
        )

        model = Baseline(
            seed=seed,
            sequence_length=sequence_length,
            num_feature=5,
            hidden_layer=hidden_layer,
            hidden_feature=hidden_feature,
            activation=activation
        )
        trainer = Trainer(max_epochs=120, gpus=-1, logger=comet_logger)
        trainer.fit(model)

    # trainer.test(model)


if __name__ == '__main__':
    main()
