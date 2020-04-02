# This file contains code for loading any dataset that
# conforms to the standard csv ('text', 'label') column
# format
from torchtext.data import TabularDataset, LabelField
import torch
import pandas as pd


class CustomDataset:
    def __init__(self, text_field, path_to_datadir: str = "../.data/custom_data", stratified_sampling=False):
        """

        @param text_field: Object of the torchtext.data.Field type used to load in the text
        from the dataset

        @param path_to_datadir: path to the directory containing the train, (val), and test files
        """
        self.text_field = text_field
        self.path_to_datadir = path_to_datadir
        self.stratified_sampling = stratified_sampling

    def load(self, targets=('label', )):
        # Just load all the columns from the csv and
        # make the fields based on this, will take
        # will take a bit more time but is easier in the
        # end
        dset_row = [("id", None), ("text", self.text_field), ("label", LabelField(dtype=torch.long))]

        train, test = TabularDataset.splits(
            path=self.path_to_datadir,
            train="train.csv",
            test="test.csv",
            format="csv",
            fields=dset_row,
            skip_header=True,
            csv_reader_params={"delimiter": ",", "quotechar": "|"}
        )
        return train, test





