from torchtext.data import Dataset, Field, TabularDataset, LabelField, Example
from typing import List, Any
import torch
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from codebase.data_classes.customdataloadermultitask import CustomDataLoaderMultiTask

# TODO: Fix the enron dataset into train, val, test


def stratified_sampler(train, test, target, text_field, label_field):
    shuffler = StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=0.30)
    X = []
    y = []
    fields = [('text', text_field), (target, label_field)]

    for example in train:
        X.append(getattr(example, "text"))
        y.append(getattr(example, target))

    for example in test:
        X.append(getattr(example, "text"))
        y.append(getattr(example, target))

    train_idx, test_idx = list(shuffler.split(X, y))[0]
    trn = Dataset(examples=[Example.fromlist([X[i], y[i]], fields) for i in train_idx], fields=fields)
    tst = Dataset(examples=[Example.fromlist([X[i], y[i]], fields) for i in test_idx], fields=fields)

    return trn, tst


class EnronDataset:
    def __init__(self, text_field, path_to_datadir: str = "../.data/enron", stratified_sampling=False):
        """

        @param text_field: Object of the torchtext.data.Field type used to load in the text
        from the dataset

        @param path_to_datadir: path to the directory containing the train, (val), and test files
        """
        self.text_field = text_field
        self.path_to_datadir = path_to_datadir
        self.stratified_sampling = stratified_sampling

    def load(self, targets=('category', 'emotion')):

        dset_row = (("id", None),
                    ("text", self.text_field),
                    ("category", LabelField(dtype=torch.long)),
                    ("emotion", LabelField(dtype=torch.long)))

        if targets == tuple("emotion"):
            dset_row = (("id", None),
                        ("text", self.text_field),
                        ("category", None),
                        ("emotion", LabelField(dtype=torch.long)))

        if targets == tuple("category"):
            dset_row = (("id", None),
                        ("text", self.text_field),
                        ("category", LabelField(dtype=torch.long)),
                        ("emotion", None))

        train, test = TabularDataset.splits(
            path=self.path_to_datadir,  # the root directory where the data lies
            train='train.csv', test="test.csv",
            format='csv',
            skip_header=True,
            # if your csv has a header, make sure to pass this to ensure it doesn't get proceesed as data!
            fields=dset_row)

        if self.stratified_sampling:
            train, test = stratified_sampler(train, test, targets, text_field=self.text_field,
                                  label_field=LabelField(dtype=torch.long))

        return train, test, targets

