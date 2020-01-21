# This file contains a simple class that generates a synthetic dataset for
# basic functionality testing of models

import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset


class SynthData(Dataset):
    # add functionality for batching
    def __init__(self,  num_points, num_features, num_classes):
        self.num_points = num_points
        self.num_features = num_features
        self.X = np.random.rand(num_points, num_features)
        self.y = np.random.randint(low=0, high=num_classes, size=num_points)
        self.num_classes = num_classes

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        # For selecting batches of items
        if torch.is_tensor(idx):
            idx = idx.tolist()


