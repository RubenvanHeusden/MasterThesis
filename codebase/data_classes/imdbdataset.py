from torchtext.datasets import IMDB
from torchtext.data import Dataset, Field
from typing import List, Any


class IMDBDataset:

    def __init__(self, text_field: Field, label_field: Field, path: str = None) -> None:
        """
        @param text_field: The textfield variable should be an instance of torchtext.data.Field
        and specificies the how the field containing the text should be pre-processed

        @param label_field: the labelfield variable should be an instance of torchtext.data.Field
        (or a child class of this class) and contains the preprocessing to do for the field
        containing the labels for training

        @param path: the path variable should be a string containing the path to the location of the
        IMDB dataset, when not set, the dataset will be loaded automatically
        """
        self.text_field = text_field
        self.label_field = label_field
        self.path = path

    def load(self) -> List[Dataset]:
        train, test = IMDB.splits(self.text_field, self.label_field, path=self.path)
        return [train, test]


