#TODO: add tests to test functionality of GPU / CPU
#TODO: Add functionality for testing of include_lens as I am changing this a lot currently

import unittest
import torch
import torch.nn as nn
import numpy as np
from codebase.models.multigatemixtureofexperts import MultiGateMixtureofExperts
from codebase.models.simplelstm import SimpleLSTM


class TestMultigateModel(unittest.TestCase):
    def setUp(self) -> None:
        self.n_tasks = 2
        self.n_experts = 3
        self.n_in = 128
        self.n_out_hidden = 128
        self.task_outs = [4, 8]
        self.batch_size = 64
        self.gating_networks = [nn.Linear(self.n_in, self.n_experts) for _ in range(self.n_tasks)]
        self.shared_layers = [nn.Linear(self.n_in, self.n_out_hidden) for _ in range(self.n_experts)]
        self.names = ["task%d" % n for n in range(self.n_tasks)]
        self.towers = {nn.Linear(self.n_out_hidden, out_dim): name for out_dim,
                                                            name in zip(self.task_outs, self.names)}

        self.model = MultiGateMixtureofExperts(shared_layers=self.shared_layers,
                                          gating_networks=self.gating_networks,
                                          towers=self.towers,
                                          device=torch.device('cpu'),
                                          include_lens=False,
                                          batch_size=self.batch_size, return_weights=False)

    def test_output_single_example(self):
        batch_size = 1
        self.model.batch_size = batch_size

        x = torch.rand(size=(batch_size, self.n_in))
        outputs = self.model(x, tower=self.names)
        for i in range(len(self.task_outs)):
            self.assertEqual(outputs[i].shape, torch.Size([self.task_outs[i]]))

    def test_output_batch_input(self):
        batch_size = 4
        self.model.batch_size = batch_size
        x = torch.rand(size=(batch_size, self.n_in))
        outputs = self.model(x, tower=self.names)
        for i in range(len(self.task_outs)):
            self.assertEqual(outputs[i].shape, torch.Size([batch_size, self.task_outs[i]]))

    def test_output_with_included_lengths(self):
        embedding_dim = 20
        hidden_dim = 24
        batch_size = 16
        sentence_lengths = torch.randint(low=1, high=self.n_in-1, size=[batch_size])

        vocab = torch.rand(size=(128, embedding_dim))
        use_lengths = True
        self.model.gating_networks = nn.ModuleList([SimpleLSTM(vocab, hidden_dim, self.n_experts,
                                                               device=torch.device("cpu"), use_lengths=use_lengths)
                                                    for _ in range(self.n_tasks)])

        self.model.shared_layers = nn.ModuleList([SimpleLSTM(vocab, hidden_dim, self.n_out_hidden,
                                                       device=torch.device("cpu"),
                                                       use_lengths=use_lengths) for _ in range(self.n_experts)])

        sentence_lengths = torch.randint(low=1, high=self.n_in - 1, size=[batch_size])
        x = torch.randint(low=0, high=30, size=(batch_size, self.n_in))

        outputs = self.model([x, sentence_lengths], tower=self.names)

        for i in range(len(self.task_outs)):
            self.assertEqual(outputs[i].shape, torch.Size([batch_size, self.task_outs[i]]))

    def test_weight_shapes(self):
        pass