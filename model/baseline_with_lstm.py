from pytorch_lightning import LightningModule
from pykrx import stock
from dataset import KOSPI200Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import torch
from layer import GraphConvolution
from .baseline import Baseline

batch_size = 512


class BaselineWithLSTM(Baseline):
    def __init__(self, sequence_length, num_feature, seed, activation='sigmoid', hidden_layer=10, hidden_feature=10, output_normalize='softmax'):
        super().__init__(sequence_length, num_feature, seed, activation='sigmoid', hidden_layer=10, hidden_feature=10, output_normalize='softmax')
        self.save_hyperparameters()
        self.lstm = nn.LSTM(5, num_feature, num_layers=1, batch_first=True)

    def forward(self, x):
        batch_size, adjacency_matrix_length, sequence_length, num_feature = x.shape
        x = x.reshape(-1, sequence_length, num_feature)
        h0 = torch.zeros(1, batch_size * adjacency_matrix_length, 5).type_as(x)
        c0 = torch.ones(1, batch_size * adjacency_matrix_length, 5).type_as(x)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = x.reshape(batch_size, adjacency_matrix_length, -1)

        for layer, batch_norm in zip(self.layers[:-1], self.batch_norms):
            x = layer(x, self.adjacency)
            x = batch_norm(x)
            x = self.activation_function(x)

        x = self.layers[-1](x, self.adjacency)
        x = x.reshape(batch_size, adjacency_matrix_length)
        output = self.output_normalize_function(x)
        return output
