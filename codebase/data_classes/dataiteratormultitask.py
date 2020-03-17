class DataIteratorMultiTask:

    def __init__(self, iterator, text_name: str = "text", label_name: str = "label") -> None:
        """
        @param iterator: torchtext iterator containing batches of the dataset
        @param text_name: attribute name of the text variable
        @param label_name: attribute name of the label variable
        """
        self.iterator = iterator
        self.text_name = text_name
        self.label_name = label_name

    def __iter__(self):
        if isinstance(self.label_name, tuple):
            for batch in self.iterator:
                labels = tuple(getattr(batch, label) for label in self.label_name)
                yield (getattr(batch, self.text_name), *labels, self.label_name)
        else:
            for batch in self.iterator:
                yield getattr(batch, self.text_name), getattr(batch, self.label_name), self.label_name

    def __len__(self) -> int:
        return len(self.iterator)
