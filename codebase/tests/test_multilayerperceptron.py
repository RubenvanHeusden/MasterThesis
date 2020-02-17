import unittest
import torch
import torch.nn as nn
from codebase.models.mlp import MLP


class TestMultiLayerPerceptron(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dim = 32
        self.hidden_dim = [16]
        self.output_dim = 3
        self.model = MLP(input_dim=self.input_dim,
                         layer_sizes=self.hidden_dim,
                         output_dim=self.output_dim)

    def test_output_single_example(self):
        x = torch.rand(size=[self.input_dim])
        x = self.model(x)
        self.assertEqual(x.shape, torch.Size([self.output_dim]))

    def test_output_batches_input(self):
        batch_size = 32
        x = torch.rand(size=(batch_size, self.input_dim))
        x = self.model(x)
        self.assertEqual(x.shape, torch.Size([batch_size, self.output_dim]))

    def test_multiple_hidden_layers(self):
        self.hidden_dim = [2, 3, 4, 5]
        self.model = MLP(input_dim=self.input_dim,
                         layer_sizes=self.hidden_dim,
                         output_dim=self.output_dim)
        batch_size = 32
        x = torch.rand(size=(batch_size, self.input_dim))
        x = self.model(x)
        self.assertEqual(x.shape, torch.Size([batch_size, self.output_dim]))