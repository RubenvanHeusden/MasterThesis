import torch
import unittest
from codebase.models.positionalencoder import PositionalEncoder


class TestPositionalEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder_model = PositionalEncoder(max_seq_len=16, embedding_dim=30, dropout=0.1, batch_first=False)

    def test_output_single_example(self):
        x = torch.rand(size=(16, 1, 30))
        out = self.encoder_model(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_batch_examples(self):
        x = torch.rand(size=(16, 8, 30))
        out = self.encoder_model(x)
        self.assertEqual(out.shape, x.shape)

    def test_batch_first_input(self):
        self.encoder_model.batch_first = True
        x = torch.rand(size=(8, 16, 30))
        out = self.encoder_model(x)
        self.assertEqual(out.shape, torch.Size([16, 8, 30]))

    def test_embedding_dim_size_assert(self):
        x = torch.rand(size=(16, 8, 29))

        with self.assertRaisesRegex(AssertionError, 'the embedding size is not correct'):
            self.encoder_model(x)

    def test_seq_len_size_assert(self):
        x = torch.rand(size=(8, 16, 30))

        with self.assertRaisesRegex(AssertionError, 'max_seq_len is of incorrect size for batch_first = False.'):
            self.encoder_model(x)

    def test_batch_first_size_assert(self):
        self.encoder_model.batch_first = True
        x = torch.rand(size=(16, 8, 30))
        with self.assertRaisesRegex(AssertionError, 'max_seq_len is of incorrect size for batch_first = True.'):
            self.encoder_model(x)
