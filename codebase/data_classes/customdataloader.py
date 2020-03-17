from torchtext.data import BucketIterator, Iterator
from codebase.data_classes.dataiterator import DataIterator
from typing import List, Any
import torch


class CustomDataLoader:
    def __init__(self, data_splits, text_field, label_field, task_name=None):
        """

        @param data_splits: a list of Dataset objects that should be turned into iterators
        @param text_field: an object of type Label to specify the pre-processing for the text field
        @param label_field: an object of type Label to specify the pre-processing for the label field
        """
        self.data_splits = data_splits
        self.text_field = text_field
        self.label_field = label_field
        self.task_name = task_name

    def construct_iterators(self, vectors: str, vector_cache: str, batch_size: int, device) -> List[Any]:
        """
        @param vectors: a string containing the type of vectors to be used
        (see https://torchtext.readthedocs.io/en/latest/vocab.html for possible options)
        @param vector_cache: string containing the lowcation of the vectors
        @param batch_size: integer specifying the size of the batches
        @param device: torch.Device indicating whether to run on CPU / GPU
        @return: list containing the iterators for train, eval and (test)
        """

        iterators = []
        # Build the vocabulary for the data, this converts all the words into integers
        # pointing to the corresponding rows in the word embedding matrix
        self.text_field.build_vocab(self.data_splits[0], vectors=vectors, vectors_cache=vector_cache,
                                    unk_init=lambda x: torch.rand(1, 300))

        # TODO: see how to remove this line, think its wrong
        self.label_field.build_vocab(self.data_splits[0])

        # Construct and iterator specifically for training
        train_iter = BucketIterator(
            self.data_splits[0],
            batch_size=batch_size,
            device=device,
            sort_within_batch=False,
            sort_key=lambda a: len(a.comment_text),
            repeat=False
        )

        iterators.append(DataIterator(train_iter, task_name=self.task_name))
        # For the other iterators (either val or test, val) construct a standard iterator with
        # appropriate settings for evaluation and test sets
        for x in range(len(self.data_splits[1:])):
            iter = Iterator(self.data_splits[x+1], batch_size=batch_size,
                         device=device, sort=False,
                         sort_within_batch=False,
                         repeat=False)
            iterators.append(DataIterator(iter, task_name=self.task_name))
        return iterators
