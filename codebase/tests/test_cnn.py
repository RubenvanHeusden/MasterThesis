from codebase.models.convnet import ConvNet
import torch.nn as nn
import numpy as np
import unittest
import torch


class TestCNN(unittest.TestCase):
    def setUp(self) -> None:
        self.sen_len = 100
        self.output_dim = 3
        self.vocab = torch.rand((100, 300))
        self.model = ConvNet(input_channels=1, num_filters=100, output_dim=self.output_dim,
                             embed_matrix=self.vocab, filter_list=[3, 4, 5])

    def test_single_input(self):
        batch_size = 1
        x = torch.randint(low=0, high=98, size=(batch_size, self.sen_len))
        outputs = self.model(x)
        self.assertEqual(outputs.shape, torch.Size([self.output_dim]))

    def test_batch_inputs(self):
        batch_size = 4
        x = torch.randint(low=0, high=98, size=(batch_size, self.sen_len))
        outputs = self.model(x)
        self.assertEqual(outputs.shape, torch.Size([batch_size, self.output_dim]))

    def test_model_on_gpu(self):
        batch_size = 1
        self.model = self.model.to(torch.device("cuda"))
        x = torch.randint(low=0, high=98, size=(batch_size, self.sen_len))
        x = x.to(torch.device("cuda"))
        outputs = self.model(x)
        self.assertEqual(outputs.detach().cpu().shape, torch.Size([self.output_dim]))
