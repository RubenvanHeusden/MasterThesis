import unittest
import torch
import torch.nn as nn
from codebase.models.multitasklstm import MultiTaskLSTM


class TestMultitaskLSTM(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_dim = 20
        self.hidden_dim = 16
        self.n_in = 8
        self.vocab = torch.rand(size=(32, self.embedding_dim))
        self.model = MultiTaskLSTM(self.vocab,
                                   self.hidden_dim,
                                   device=torch.device("cpu"),
                 use_lengths=False)

    def test_output_single_example(self):
        batch_size = 1
        x = torch.randint(low=0, high=10, size=(batch_size, self.n_in))
        x = self.model(x)
        self.assertEqual(x.shape, torch.Size([batch_size, self.hidden_dim]))

    def test_output_batches_input(self):
        batch_size = 4
        x = torch.randint(low=0, high=10, size=(batch_size, self.n_in))
        x = self.model(x)
        self.assertEqual(x.shape, torch.Size([batch_size, self.hidden_dim]))

    def test_input_with_lengths_included(self):
        self.model.use_lengths = True
        batch_size = 4
        sequence_length = 16
        sentence_lengths = torch.randint(low=1, high=sequence_length-1, size=[batch_size])
        x = torch.randint(low=0, high=sequence_length-1, size=(batch_size, sequence_length))
        x = self.model((x, sentence_lengths))
        self.assertEqual(x.shape, torch.Size([batch_size, self.hidden_dim]))
