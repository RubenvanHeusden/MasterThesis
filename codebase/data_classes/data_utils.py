import torch
import itertools
from typing import List, Any
from torchtext.data import Field, LabelField, Example, TabularDataset, Dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedShuffleSplit
from codebase.data_classes.customdataloader import CustomDataLoader
from codebase.data_classes.amazondataset import AmazonDataset
from codebase.data_classes.yelpdataset import YelpDataset
from codebase.data_classes.yahoodataset import YahooDataset
from codebase.data_classes.imdbdataset import IMDBDataset
from codebase.data_classes.sttdataset import SSTDataset
from codebase.data_classes.dailydialogdataset import DailyDialogDataset
from codebase.data_classes.enrondataset import EnronDataset
from codebase.data_classes.customdataloadermultitask import CustomDataLoaderMultiTask
import numpy as np
# TODO: Add complete functionality for new datasets

# TODO: To fix the issue with the vocab not working for multiple tasks
# change the input to construct iterators so that each entry list contains both
# datasets


def combine_datasets(list_of_dataset_classes: List[Any], include_lens: bool,
                     set_lowercase: bool, batch_size: int, task_names: List[str]):
    """

    @param list_of_dataset_classes: List containing the dataset classes that should be loaded for the multitask
    training and testing

    @param include_lens: boolean indicating whether or not to use the lengths of the original sequence to minimize
    the amount of padding needed

    @param set_lowercase: boolean indication whether or not to convert all text to lowerdase

    @param batch_size: integer specifying the batch sizes used for training and testing

    @param task_names: list of strings specifying the names of the datasets, this is used for
    saving the model parameters with the appropriate names

    @return: combined vocabulary matrix and dataloaders for the  combined datasets
    """
    # return combined iterators for train, val and test
    print("---Loading in datasets---")
    datasets = {}
    for i, dataset_class in enumerate(list_of_dataset_classes):
        TEXT = Field(lower=set_lowercase, include_lengths=include_lens, batch_first=True)
        # TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=True, batch_first=True)
        LABEL = LabelField(dtype=torch.long)
        dataset = dataset_class(TEXT, LABEL).load()
        # Load the IMDB dataset and split it into train and test portions
        dloader = CustomDataLoader(dataset, TEXT, LABEL, task_name=task_names[i])
        data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                                             batch_size=batch_size, device=torch.device("cpu"))
        datasets[task_names[i]] = {'text': TEXT, 'label': LABEL, 'iters': data_iterators}

    total_vocab = torch.cat(tuple(datasets[model]['text'].vocab.vectors for model in datasets.keys()))
    train_iterators = itertools.chain(*zip(*tuple(datasets[model]['iters'][0] for model in datasets.keys())))
    test_iterators = list(datasets[model]['iters'][-1] for model in datasets.keys())
    print("---Finished Loading in datasets---")
    return total_vocab, list(train_iterators), test_iterators


def single_task_dataset_prep(dataset_string):
    if dataset_string == "DAILYDIALOG-EMOT":
        dataset = DailyDialogDataset
        output_dim = 7
        target = "emotion"

    elif dataset_string == "DAILYDIALOG-TOPIC":
        dataset = DailyDialogDataset
        output_dim = 10
        target = "topic"

    elif dataset_string == "DAILYDIALOG-ACT":
        dataset = DailyDialogDataset
        output_dim = 4
        target = "act"

    elif dataset_string == "ENRON-CAT":
        dataset = EnronDataset
        output_dim = 6
        target = "category"

    elif dataset_string == "ENRON-EMOT":
        dataset = EnronDataset
        output_dim = 10
        target = "emotion"

    elif dataset_string == "SST":
        dataset = SSTDataset
        output_dim = 5
        target = "label"
    else:
        raise(Exception("Invalid dataset argument, please refer to the help function of the "
                        "argument parser for details on valid arguments"))
    return dataset, output_dim, target


def multi_task_dataset_prep(dataset_string):
    if dataset_string == "DAILYDIALOG":
        dataset = DailyDialogDataset
        output_dim = [7, 4, 10]
        targets = ("emotion", "act", "category")

    elif dataset_string == "ENRON":
        dataset = EnronDataset
        output_dim = [6, 20]
        targets = ("category", "emotion")

    else:
        raise(Exception("Invalid dataset argument, please refer to the help function of the "
                        "argument parser for details on valid arguments"))
    return dataset, output_dim, targets




def oversampler(train, test, target, text_label, field_label):
    X = np.array(range(len(train))).reshape(-1, 1)
    y = []
    sampler = RandomOverSampler(random_state=0)
    for example in train:
        y.append(getattr(example, target))
    resampled_X, resampled_Y = sampler.fit_resample(X, y)
    for text, label in zip(resampled_X, resampled_Y):
        print(text, label)

