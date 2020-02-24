import unittest
import torch
import torch.nn as nn
from codebase.models.simplelstm import SimpleLSTM


class TestSimpleLSTMmodel(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_dim = 4
        self.vocab = torch.rand(size=(32, self.embedding_dim))
        self.hidden_dim = 16
        self.output_dim = 5
        self.model = SimpleLSTM(self.vocab, self.hidden_dim, self.output_dim, dropout=0.3,
                                device=torch.device("cpu"), use_lengths=False)

    def test_output_single_example(self):
        batch_size = 1
        sequence_length = 20
        x = torch.randint(low=0, high=sequence_length-1, size=(batch_size, sequence_length))
        x = self.model(x)
        self.assertEqual(x.shape, torch.Size([batch_size, self.output_dim]))

    def test_output_batches_examples(self):
        batch_size = 8
        sequence_length = 20
        x = torch.randint(low=0, high=sequence_length-1, size=(batch_size, sequence_length))
        x = self.model(x)
        self.assertEqual(x.shape, torch.Size([batch_size, self.output_dim]))

    def test_output_with_included_sequences_lengths(self):
        self.model.use_lengths = True
        batch_size = 4
        sequence_length = 16
        sentence_lengths = torch.randint(low=1, high=sequence_length-1, size=[batch_size])
        x = torch.randint(low=0, high=sequence_length-1, size=(batch_size, sequence_length))
        x = self.model((x, sentence_lengths))
        self.assertEqual(x.shape, torch.Size([batch_size, self.output_dim]))

    def test_gpu_tensors(self):
        batch_size = 4
        sequence_length = 16
        x = torch.randint(low=0, high=sequence_length-1, size=(batch_size, sequence_length)).cuda()
        self.model.device = torch.device("cuda")
        self.model.to(torch.device("cuda"))
        x = self.model(x)
        self.assertIsInstance(x, torch.cuda.FloatTensor)

