from pytorch_lightning import LightningModule
from pykrx import stock
from dataset import KOSPI200Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import torch
from layer import GraphConvolution

batch_size = 512


class Baseline(LightningModule):
    def __init__(self):
        super().__init__()
        self.adjacency = nn.Parameter(torch.rand(200, 200) + torch.eye(200), True)
        self.gc1 = GraphConvolution(5, 10)
        # self.gc2 = GraphConvolution(10, 10)
        self.gc3 = GraphConvolution(10, 5)
        self.gc4 = GraphConvolution(5, 1)

    def forward(self, x):
        batch_size, adjacency_matrix_length, num_feature = x.shape
        x = F.sigmoid(self.gc1(x, self.adjacency))
        # x = F.sigmoid(self.gc2(x, self.adjacency))
        x = F.sigmoid(self.gc3(x, self.adjacency))
        x = self.gc4(x, self.adjacency)
        # print(self.adjacency)
        x = x.reshape(batch_size, adjacency_matrix_length)
        # print(x[0])
        softmax = F.softmax(x, dim=1)
        # print(softmax[0])
        return softmax

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def prepare_data(self) -> None:
        KOSPI200Dataset.load_data()
        KOSPI200Dataset.process_data()

    def _share_step(self, batch, batch_idx):
        x, y = batch
        softmax = self(x)
        profit = y[:, :, 3] / (y[:, :, 0] + 1e-8)
        loss = -torch.mean(torch.log(torch.sum(softmax * profit, dim=1)))
        return loss

    def train_dataloader(self) -> DataLoader:
        train_dataset = KOSPI200Dataset()
        return DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        loss = self._share_step(batch, batch_idx)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    # validation
    def val_dataloader(self):
        val_dataset = KOSPI200Dataset(mode='val')
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def validation_step(self, batch, batch_idx):
        loss = self._share_step(batch, batch_idx)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = -torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'profit': torch.exp(avg_loss)}
        return {'log': logs}

    # 테스트
    def test_dataloader(self):
        test_dataset = KOSPI200Dataset(mode='test')
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        softmax = self(x)
        profit = y[:, :, 3] / (y[:, :, 0] + 1e-8)
        profit = torch.log(torch.sum(softmax * profit, dim=1))
        accumulator = 0
        for i, v in enumerate(profit):
            accumulator = accumulator + v
            self.logger.log_metrics({'test_daily_profit': torch.exp(v)}, i + 1)
            self.logger.log_metrics({'test_accumulate_profit': torch.exp(accumulator)}, i + 1)

        loss = -torch.mean(profit)
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = -torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'profit': torch.exp(avg_loss)}
        return {'log': logs}

