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
from os import path
import hydra.utils

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
        return_rate = y[:, :, 3] / (y[:, :, 0] + 1e-8)
        return_rate[return_rate == 0] = 1  # 거래정지/상장폐지 등으로 거래 불가능한 것들 투자결정으로 loss 오염되지 않게 보정
        profit = output * return_rate * 0.997
        profit = torch.mean(torch.log(torch.sum(profit, dim=1)))  # 수수료&거래세&유관기관 => 수수료 0.3%
        value, indices = torch.max(output, dim=1)
        loss = - profit + 0.8 * torch.mean(value)
        return loss, torch.exp(profit)

    def train_dataloader(self) -> DataLoader:
        train_dataset = KOSPI200Dataset()
        return DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        loss, profit = self.calculate_loss(batch, batch_idx)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    # validation
    def val_dataloader(self):
        val_dataset = KOSPI200Dataset(mode='val')
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def validation_step(self, batch, batch_idx):
        loss, profit = self.calculate_loss(batch, batch_idx)
        return {'loss': loss, 'profit': profit}

    def validation_epoch_end(self, outputs):
        profit = torch.stack([x['profit'] for x in outputs])
        profit = torch.exp(torch.log(profit).mean())
        logs = {'profit': profit}
        return {'log': logs}

    # 테스트
    def test_dataloader(self):
        test_dataset = KOSPI200Dataset(mode='val') # validation 에서의 일 수익률을 확인하고 싶어
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        return_rate = y[:, :, 3] / (y[:, :, 0] + 1e-7)

        accumulator = 0
        for i in range(len(output)):
            output_element = output[i]
            self.visualize_to_comet(output_element, 'blue', f'{"%02d" % (i + 1)}_daily_decision')
            return_rate_element = return_rate[i]
            self.visualize_to_comet(return_rate_element, 'green', f'{"%02d" % (i + 1)}_return_rate')
            return_rate_element[return_rate_element == 0] = 1  # 거래정지/상장폐지 등으로 거래 불가능한 것들 투자결정으로 loss 오염되지 않게 보정
            profit_element = output_element * return_rate_element * 0.997
            self.visualize_to_comet(profit_element - output_element, 'red', f'{"%02d" % (i + 1)}_profit_result')

            v = torch.log(torch.sum(profit_element, dim=0))  # 수수료&거래세&유관기관 => 수수료 0.3%
            accumulator = accumulator + v

            self.logger.log_metrics({'test_daily_profit': torch.exp(v)}, i + 1)
            self.logger.log_metrics({'test_accumulate_profit': torch.exp(accumulator)}, i + 1)

        return None

    def test_epoch_end(self, outputs):
        # avg_loss = -torch.stack([x['loss'] for x in outputs]).mean()
        # logs = {'profit': torch.exp(avg_loss)}
        # return {'log': logs}
        return None

    def visualize_to_comet(self, weights, color, name):
        index = np.arange(len(weights))
        fig = plt.figure()
        plt.hist(index, weights=weights.cpu(), bins=len(weights), color=color)
        pil = fig2img(fig)
        plt.close(fig)
        self.logger.experiment.log_image(pil, name=name, overwrite=True, image_scale=2.0)

