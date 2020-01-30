import torch
from typing import Dict
from torch.utils.data import Dataset

# This file contains a simple class that generates a synthetic dataset for
# basic functionality testing of models


class SynthData(Dataset):
    # add functionality for batching
    def __init__(self,  num_points: int, num_features: int, num_classes: int) -> None:
        """
        @param num_points: integer specifying the number of points the dataset should contain
        @param num_features: integer specifying the number of features of each datapoint
        @param num_classes: integer specifying the number of possible classes
        """
        self.num_points = num_points
        self.num_features = num_features
        self.X = torch.rand(num_points, num_features)
        self.y = torch.randint(high=num_classes, size=(num_points, 1))
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_points

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {"X": self.X[idx], "y": self.y[idx]}
