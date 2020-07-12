from torchtext.data import BucketIterator, Iterator
from codebase.data_classes.dataiterator import DataIterator

# This class implements a dataloader class designed for loading
# in a dataset that contains multiple distinct labels for each data point
# (for example, intent and emotion)

# The datapoints are loaded in separately with one label each time and an appropriate task
# associated with it

# This takes the field names in the right order as input, so that the dataloader
# can return tuples of the right shape


class CustomDataLoader:
    def __init__(self, datasplits, text_field, field_names):
        """
        @param text_field: Object of the torchtext.data.Field type used to load in the text
        from the dataset
        """
        self.datasplits = datasplits
        self.text_field = text_field
        self.field_names = field_names

    def construct_iterators(self, vectors: str, vector_cache: str, batch_size: int, device):
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
        self.text_field.build_vocab(self.datasplits[0])
        self.text_field.build_vocab(self.datasplits[0], vectors=vectors, vectors_cache=vector_cache)
        for key, val in self.datasplits[0].fields.items():
            if key != "text" and key != 'id' and val:
                val.build_vocab(self.datasplits[0])
        # Construct an iterator specifically for training
        train_iter = BucketIterator(
            self.datasplits[0],
            batch_size=batch_size,
            device=device,
            sort_within_batch=False,
            sort_key=lambda a: len(a.text),
            repeat=False
        )

        iterators.append(DataIterator(train_iter, label_name=self.field_names))

        for x in range(len(self.datasplits[1:])):
            iter = Iterator(self.datasplits[x+1], batch_size=batch_size,
                         device=device, sort=False,
                         sort_within_batch=False,
                         repeat=False)
            iterators.append(DataIterator(iter, label_name=self.field_names))
        return iterators



