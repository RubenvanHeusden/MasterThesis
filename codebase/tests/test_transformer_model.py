import torch
import unittest
from codebase.models.transformermodel import TransformerModel


class TestTransformerModel(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_dim = 50
        self.max_seq_len = 16
        self.num_outputs = 3
        embedding_matrix = torch.rand(size=(20, self.embedding_dim))
        self.transformer_model = TransformerModel(max_seq_len=self.max_seq_len,
                                                  num_outputs=self.num_outputs,
                                                  word_embedding_matrix=embedding_matrix,
                                                  feed_fwd_dim=4,
                                                  num_transformer_layers=1,
                                                  num_transformer_heads=2,
                                                  batch_first=False,
                                                  pad_index=-1)

    def test_single_example_output(self):
        sentence = torch.randint(low=0, high=19, size=[16, 1])
        out = self.transformer_model(sentence)
        self.assertEqual(out.shape, torch.Size([1, self.num_outputs]))

    def test_batch_examples_output(self):
        batch_size = 4
        sentence = torch.randint(low=0, high=19, size=[16, batch_size])
        out = self.transformer_model(sentence)
        self.assertEqual(out.shape, torch.Size([batch_size, self.num_outputs]))

    def test_batch_first_output(self):
        self.transformer_model.batch_first = True
        self.transformer_model.positional_encoder.batch_first = True
        batch_size = 4
        sentence = torch.randint(low=0, high=19, size=[batch_size, 16])
        out = self.transformer_model(sentence)
        self.assertEqual(out.shape, torch.Size([batch_size, self.num_outputs]))

    def test_assert_input_shape(self):
        x = torch.rand(size=(18, self.max_seq_len))
        with self.assertRaisesRegex(AssertionError, " max_seq_len is of incorrect size for batch_first = False."):
            self.transformer_model(x)

    def test_assert_batch_first_input_shape(self):
        self.transformer_model.batch_first = True
        self.transformer_model.positional_encoder.batch_first = True
        x = torch.rand(size=(self.max_seq_len, 18))
        with self.assertRaisesRegex(AssertionError, " max_seq_len is of incorrect size for batch_first = True."):
            self.transformer_model(x)

    def test_assert_padding_mask(self):
        pass

    def test_assert_cls_token(self):
        pass
