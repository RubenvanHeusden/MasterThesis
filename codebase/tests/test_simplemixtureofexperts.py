import unittest
import torch
import torch.nn as nn
from codebase.models.simplemoe import SimpleMoE
from codebase.models.simplelstm import SimpleLSTM

class TestSimpleMixtureofExperts(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_dim = 300
        self.num_experts = 4
        self.num_outs = 5
        self.hidden_dim = 64

        self.vocab = torch.rand(size=(128, self.embedding_dim))
        self.gating_network = SimpleLSTM(vocab=self.vocab, embedding_dim=self.embedding_dim,
                                         hidden_dim=self.hidden_dim, output_dim=self.num_experts,
                                         device=torch.device("cpu"), use_lengths=False)

        self.exerts_networks = [SimpleLSTM(vocab=self.vocab, embedding_dim=self.embedding_dim,
                                         hidden_dim=self.hidden_dim, output_dim=self.num_outs,
                                         device=torch.device("cpu"),
                                           use_lengths=False
                                          ) for _ in range(self.num_experts)]

        self.model = SimpleMoE(self.gating_network, self.exerts_networks, output_dim=self.num_outs,
                               device=torch.device("cpu"))

    def test_output_single_example(self):
        batch_size = 1
        sequence_length = 128
        x = torch.randint(low=0, high=126, size=(batch_size, sequence_length))
        x = self.model(x)
        self.assertEqual(x[0].shape, torch.Size([self.num_outs]))

    def test_output_batches(self):
        batch_size = 8
        sequence_length = 32
        x = torch.randint(low=0, high=30, size=(batch_size, sequence_length))
        x = self.model(x)
        self.assertEqual(x[0].shape, torch.Size([batch_size, self.num_outs]))


    def test_model_with_included_lengths(self):
        pass

    def test_model_on_gpu(self):
        pass