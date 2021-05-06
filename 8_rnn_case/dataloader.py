import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import datetime
import pickle


def normalization(data, max, min):
    return (data - min) / (max - min)


def standardization(data, mean, std):
    return (data - mean) / std


def reconstruct(data, mean, std):
    return data * std + mean


class DataLoader:
    def __init__(self, data_type, filename, augment_test_data=True):
        self.augment_test_data = augment_test_data
        self.X_train, self.y_train = self.preprocessing(Path('data', 'train', filename), train=True)
        self.X_test, self.y_test = self.preprocessing(Path('data', 'test', filename), train=False)

    def augmentation(self, data, label, noise_ratio=0.05,
                     noise_interval=0.0005, max_length=100000):
        noise_seq = torch.randn(data.size())  # 标准正态分布
        augmentaed_data = data.clone()
        augmentaed_label = label.clone()
        for i in np.arange(0, noise_ratio, noise_interval):
            scaled_noise_seq = noise_ratio * self.std.expand_as(data) * noise_seq
            augmentaed_data = torch.cat([augmentaed_data, data+scaled_noise_seq], dim=0)
            augmentaed_label = torch.cat([augmentaed_label, label])
            if len(augmentaed_data) > max_length:
                augmentaed_data = augmentaed_data[:max_length]
                augmentaed_label = augmentaed_label[:max_length]
                break

        return augmentaed_data, augmentaed_label

    def preprocessing(self, path, train=True):
        with open(str(path), 'rb') as f:
            data = torch.FloatTensor(pickle.load(f))
            label = data[:, -1]
            data = data[:, :-1]

        if train:
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
            self.length = len(data)
            data, label = self.augmentation(data, label)
        else:
            if self.augment_test_data:
                data, label = self.augmentation(data, label)

        data = standardization(data, self.mean, self.std)

        return data, label

    def batchify(self, args, data, batch_size):
        n = data.size(0) // batch_size
        trimmed_data = data.narrow(0, 0, n * batch_size)  # 表示取变量x的第dimension维,从索引start开始到,start+length范围的值
        # contiguous配合view使用，使内存顺序改变；transpose交换0维和1维
        # (时间步，batch，维度)
        batched_data = trimmed_data.contiguous().view(batch_size, -1, trimmed_data.size(-1)).transpose(0, 1)
        batched_data = batched_data.to('cpu')

        return batched_data


