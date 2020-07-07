from torch.utils.data import Dataset
from data.kospi200_metadata import _2018_01_02
from pykrx import stock
import time
import pandas as pd
from os import path
import os
import numpy as np


class KOSPI200Dataset(Dataset):
    data = None
    max_length = 0
    train_length = 2000
    val_length = 500
    test_length = 500
    sequence_length = 1

    @classmethod
    def metadata(cls):
        kospi200 = _2018_01_02
        codes = list(map(lambda code: '%06d' % code, kospi200.keys()))
        names = {}
        for k, v in kospi200.items():
            names['%06d' % k] = v
        return codes, names

    # TODO
    # kospi 구성 종목을 원하는 시간 대의 구성 종목으로 설정할 수 있도록 파라미터 추가
    @classmethod
    def download_data(cls, save_dir='./data/kospi'):
        os.makedirs(save_dir, exist_ok=True)

        codes, names = cls.metadata()
        for code in codes:
            df = stock.get_market_ohlcv_by_date('19000101', '20300101', code)
            df.to_csv(path.join(save_dir, f'{names[code]}_{code}.csv'))
            time.sleep(1)

    @classmethod
    def load_data(cls, save_dir='./data/kospi'):
        codes, names = cls.metadata()
        data = {}
        for code in codes:
            data[code] = pd.read_csv(path.join(save_dir, f'{names[code]}_{code}.csv'), index_col=0).to_numpy(dtype=np.float32)
            cls.max_length = cls.max_length if cls.max_length >= data[code].shape[0] else data[code].shape[0]
        cls.data = data

    # 1. 아직 상장되어 있지 않아서 데이터가 없는 경우에는 0 으로 padding
    # 2. 거래 정지 때문에 거래량이 0 인 경우, 그냥 다 0 으로 처리
    # TODO
    # 상장폐지된 종목의 경우, padding 처리 과정에서 버그 발생
    # sequence 으로 데이터를 만들려면 normalize 로직에 대해서 고민해 봐야함
    @classmethod
    def process_data(cls):
        codes, names = cls.metadata()
        for code in codes:
            length = cls.data[code].shape[0]
            cls.data[code] = np.pad(cls.data[code], ((cls.max_length - length, 0), (0, 0)), 'constant', constant_values=0)
            cls.data[code][cls.data[code][:, 4] <= 0] = 0

    @classmethod
    def setup(cls, train_length, val_length, test_length, sequence_length=1):
        cls.train_length = train_length
        cls.val_length = val_length
        cls.test_length = test_length
        cls.sequence_length = sequence_length

    def __init__(self, mode='train'):
        super().__init__()
        self.mode = mode

    def __getitem__(self, index):
        adj_index = None
        if self.mode == 'train':
            adj_index = -(self.train_length + self.val_length + self.test_length) + index - self.sequence_length
        if self.mode == 'val':
            adj_index = -(self.val_length + self.test_length) + index - self.sequence_length
        if self.mode == 'test':
            adj_index = -self.test_length + index - self.sequence_length

        codes, names = self.metadata()
        x = []
        y = []
        for code in codes:
            sequence = []
            for i in range(self.sequence_length):
                if self.data[code][adj_index - i][0] <= 0:
                    x_price = self.data[code][adj_index - i][0:4]
                    x_volume = self.data[code][adj_index - i][4:5]
                else:
                    x_price = self.data[code][adj_index - i][0:4] / self.data[code][adj_index - i][0]
                    x_volume = self.data[code][adj_index - i][4:5]
                sequence.append(np.concatenate([x_price, x_volume]))

            if self.data[code][adj_index + 1][0] <= 0:
                y_price = self.data[code][adj_index + 1][0:4]
                y_volume = self.data[code][adj_index + 1][4:5]
            else:
                y_price = self.data[code][adj_index + 1][0:4] / self.data[code][adj_index + 1][0]
                y_volume = self.data[code][adj_index + 1][4:5]

            x.append(np.stack(sequence, axis=0))
            y.append(np.concatenate([y_price, y_volume]))

        return np.stack(x, axis=0), np.stack(y, axis=0)

    def __len__(self):
        if self.mode == 'train':
            return self.train_length
        if self.mode == 'val':
            return self.val_length
        if self.mode == 'test':
            return self.test_length
