"""
This file contains code for loading any dataset that
conforms to the standard csv ('text', 'label') column
format
"""

import torch
import pandas as pd
from torchtext.data import Example, Dataset
from torchtext.data import TabularDataset, LabelField
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_sampler(train, test, target, text_field, label_field):
    shuffler = StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=0.30)
    X = []
    y = []
    fields = [('text', text_field), (target[0], label_field)]

    for example in train:
        X.append(getattr(example, "text"))
        y.append(getattr(example, target[0]))

    for example in test:
        X.append(getattr(example, "text"))
        y.append(getattr(example, target[0]))

    train_idx, test_idx = list(shuffler.split(X, y))[0]

    trn = Dataset(examples=[Example.fromlist([X[i], y[i]], fields) for i in train_idx], fields=fields)
    tst = Dataset(examples=[Example.fromlist([X[i], y[i]], fields) for i in test_idx], fields=fields)

    return trn, tst


class CSVDataset:
    def __init__(self, text_field, path_to_datadir, stratified_sampling=False):
        """
        :param text_field: Object of the torchtext.data.Field type used to load in the text \
        from the dataset
        :param path_to_datadir: path to the directory containing the train, (val), and test files
        """
        self.text_field = text_field
        self.path_to_datadir = path_to_datadir
        self.stratified_sampling = stratified_sampling

    def load(self, text_label_col: str = "text", targets=('label', ), delimiter: str = ",", quotechar: str = '"'):
        field_headers = list(pd.read_csv(self.path_to_datadir+"train.csv", quotechar=quotechar, sep=delimiter))
        dset_row = []
        for header in field_headers:
            if header == text_label_col:
                dset_row.append((text_label_col, self.text_field))
            elif header in targets:
                dset_row.append((header, LabelField(dtype=torch.long)))
            else:
                dset_row.append((header, None))

        train, test = TabularDataset.splits(
            path=self.path_to_datadir,
            train='train.csv', test="test.csv",
            format='csv',
            skip_header=True,
            fields=dset_row)

        if self.stratified_sampling:
            train, test = stratified_sampler(train, test, targets, text_field=self.text_field,
                                             label_field=LabelField(dtype=torch.long))
        return train, test

