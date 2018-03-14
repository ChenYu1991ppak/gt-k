import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from utils import find_files, read_file

mean_std = [(2.3975068019282095, 0.74820653747927679), (0.76617039203917292, 0.49363755065411041), (3.4163245332664989,
            0.9517764586294204), (8.0389186170805775, 1.0525454629575819), (7.9106944792142366, 1.7287003808959538),
            (0.54898590834333261, 0.18216682120141184), (0.84683352600386297, 0.12355401794240123), (8.535008819904915,
            1.0499958547971018), (0.33121660717131385, 0.27557020851377778), (10.267847808480052, 2.2258863523932733),
            (14.410284581780207, 3.2578219718129966), (26.061836149358449, 8.7218866926313581), (7.0583783511675904, 2.7074727980463864),
            (10.405868200915997, 2.9959432910477561), (5.6165793445323402, 2.1699724845728392), (6.9886332742197412, 1.2240518597057848)]

def normalize_image(list):
    data = np.array(list)
    for i in range(16):
        data[i] = (data[i] - mean_std[i][0]) / mean_std[i][1]
    matrix = data.reshape(4, 4, 1)
    matrix = matrix.transpose(2, 0, 1)
    return matrix

class TrainDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, item):
        track, label = self.samples[item][0], self.samples[item][1]
        track = normalize_image(track)
        return track, label

    def __len__(self):
        return len(self.samples)

class TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, item):
        track, label = self.samples[item][0], self.samples[item][1]
        track = normalize_image(track)
        return track, label

    def __len__(self):
        return len(self.samples)


def load_data(data_path):
    data_file = find_files(data_path)
    data = []
    for d in data_file:
        samples = list(read_file(d))
        data.extend(samples)
    random.shuffle(data)
    return data

def get_mean_std(data):
    mean_std = []
    temp = []
    for i in range(16):
        for d, _ in data:
            temp.append(d[i])
        vector = np.array(temp)
        mean_std.append((vector.mean(), vector.std()))
        temp = []
    return mean_std
