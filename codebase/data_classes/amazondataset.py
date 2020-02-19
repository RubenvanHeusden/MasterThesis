from torchtext.data import Dataset, Field, TabularDataset
from typing import List, Any


class AmazonDataset:
    def __init__(self, text_field: Field, label_field: Field, path: str = "../.data/amazon") -> None:
        """
        @param text_field: The textfield variable should be an instance of torchtext.data.Field
        and specificies the how the field containing the text should be pre-processed

        @param label_field: the labelfield variable should be an instance of torchtext.data.Field
        (or a child class of this class) and contains the preprocessing to do for the field
        containing the labels for training

        @param path: the path variable should be a string containing the path to the location of the
        SST dataset, when not set, the dataset will be loaded automatically
        """
        self.text_field = text_field
        self.label_field = label_field
        self.path = path

    def load(self) -> List[Dataset]:
        train, test = TabularDataset.splits(
            path=self.path,  # the root directory where the data lies
            train='train.tsv', validation="test.tsv",
            format='tsv',
            skip_header=False,
            # if your csv has a header, make sure to pass this to ensure it doesn't get proceesed as data!
            fields=(("id", None), ("label", self.label_field), ("text", self.text_field)))

        return [train, test]