# This file contains a simple class that generates a synthetic dataset for
# basic functionality testing of models

import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SynthData(Dataset):
    # add functionality for batching
    def __init__(self,  num_points, num_features, num_classes):
        self.num_points = num_points
        self.num_features = num_features
        self.X = torch.rand(num_points, num_features)
        self.y = torch.randint(high=num_classes, size=(num_points, 1))
        self.num_classes = num_classes

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return self.X[idx]
