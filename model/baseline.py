from pytorch_lightning import LightningModule
from pykrx import stock
from dataset import KOSPI200Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import torch
from layer import GraphConvolution
import matplotlib.pyplot as plt
import numpy as np
from tools import fig2img

batch_size = 512


class Baseline(LightningModule):
    def __init__(self, sequence_length, num_feature, seed, activation='sigmoid', hidden_layer=10, hidden_feature=10, output_normalize='softmax'):
        super().__init__()
        self.save_hyperparameters()

        num_input_feature = sequence_length * num_feature
        self.adjacency = nn.Parameter(torch.rand(200, 200) + torch.eye(200), True)
        layers = [GraphConvolution(num_input_feature, hidden_feature)]
        for i in range(hidden_layer):
            layers.append(GraphConvolution(hidden_feature, hidden_feature))
        layers.append(GraphConvolution(hidden_feature, 1))
        self.layers = nn.ModuleList(layers)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(200) for i in range(hidden_layer - 1)])

    def activation_function(self, x):
        if self.hparams.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.hparams.activation == 'tanh':
            return torch.tanh(x)

    def output_normalize_function(self, x):
        if self.hparams.output_normalize == 'softmax':
            return F.softmax(x, dim=1)
        elif self.hparams.output_normalize == 'ratio':
            return x / x.sum(dim=1, keepdim=True)

    def forward(self, x):
        batch_size, adjacency_matrix_length, sequence_length, num_feature = x.shape
        x = x.reshape(batch_size, adjacency_matrix_length, -1)

        for layer, batch_norm in zip(self.layers[:-1], self.batch_norms):
            x = layer(x, self.adjacency)
            x = batch_norm(x)
            x = self.activation_function(x)

        x = self.layers[-1](x, self.adjacency)
        x = x.reshape(batch_size, adjacency_matrix_length)
        output = self.output_normalize_function(x)
        return output

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def prepare_data(self) -> None:
        KOSPI200Dataset.load_data()
        KOSPI200Dataset.process_data()

    def calculate_loss(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        profit = y[:, :, 3] / (y[:, :, 0] + 1e-8)
        loss = -torch.mean(torch.log(torch.sum(output * profit * 0.997, dim=1)))  # 수수료&거래세&유관기관 => 수수료 0.3%
        return loss

    def train_dataloader(self) -> DataLoader:
        train_dataset = KOSPI200Dataset()
        return DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    # validation
    def val_dataloader(self):
        val_dataset = KOSPI200Dataset(mode='val')
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch, batch_idx)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = -torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'profit': torch.exp(avg_loss)}
        return {'log': logs}

    # 테스트
    def test_dataloader(self):
        test_dataset = KOSPI200Dataset(mode='val') # validation 에서의 일 수익률을 확인하고 싶어
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        profit = y[:, :, 3] / (y[:, :, 0] + 1e-7)

        accumulator = 0
        for i in range(len(output)):
            output_element = output[i]
            return_rate_element = profit[i]
            profit_element = output_element * return_rate_element * 0.997
            v = torch.log(torch.sum(profit_element, dim=0))  # 수수료&거래세&유관기관 => 수수료 0.3%
            accumulator = accumulator + v

            index = np.arange(len(output_element))
            codes, _ = KOSPI200Dataset.metadata()

            fig = plt.figure()
            plt.hist(index, weights=output_element.cpu(), bins=len(output_element), color='blue')
            pil = fig2img(fig)
            plt.close(fig)
            self.logger.experiment.log_image(pil, name=f'{"%02d" % (i + 1)}_daily_decision', overwrite=True, image_scale=2.0)
            fig = plt.figure()
            plt.hist(index, weights=return_rate_element.cpu(), bins=len(return_rate_element), color='green')
            pil = fig2img(fig)
            plt.close(fig)
            self.logger.experiment.log_image(pil, name=f'{"%02d" % (i + 1)}_return_rate_decision', overwrite=True, image_scale=2.0)
            fig = plt.figure()
            plt.hist(index, weights=profit_element.cpu(), bins=len(profit_element), color='red')
            pil = fig2img(fig)
            plt.close(fig)
            self.logger.experiment.log_image(pil, name=f'{"%02d" % (i + 1)}_profit_result', overwrite=True, image_scale=2.0)

            self.logger.log_metrics({'test_daily_profit': torch.exp(v)}, i + 1)
            self.logger.log_metrics({'test_accumulate_profit': torch.exp(accumulator)}, i + 1)

        loss = -torch.mean(profit)
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = -torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'profit': torch.exp(avg_loss)}
        return {'log': logs}

