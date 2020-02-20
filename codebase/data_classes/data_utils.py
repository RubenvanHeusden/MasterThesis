import torch
import itertools
from typing import List, Any
from torchtext.data import Field, LabelField
from codebase.data_classes.customdataloader import CustomDataLoader
from codebase.data_classes.amazondataset import AmazonDataset
from codebase.data_classes.yelpdataset import YelpDataset
from codebase.data_classes.yahoodataset import YahooDataset
from codebase.data_classes.imdbdataset import IMDBDataset
from codebase.data_classes.sttdataset import SSTDataset


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
    if dataset_string == "SST":
        dataset = SSTDataset
        output_dim = 2
    elif dataset_string == "YELP":
        dataset = YelpDataset
        output_dim = 5
    elif dataset_string == "IMDB":
        dataset = IMDBDataset
        output_dim = 2
    elif dataset_string == "AMAZON":
        dataset = AmazonDataset
        output_dim = 5
    elif dataset_string == "YAHOO":
        dataset = YahooDataset
        output_dim = 10
    else:
        raise(Exception("Invalid dataset argument, please refer to the help function of the "
                        "argument parser for details on valid arguments"))
    return dataset, output_dim

def multi_task_dataset_prep(list_of_dataset_strings):
    output_dimensions = []
    dataset_names = []
    datasets = []
    for dset in list_of_dataset_strings:
        if dset == "SST":
            dataset_names.append("SST")
            datasets.append(SSTDataset)
            output_dimensions.append(2)
        elif dset == "YELP":
            dataset_names.append("YELP")
            datasets.append(YelpDataset)
            output_dimensions.append(5)
        elif dset == "IMDB":
            dataset_names.append("IMDB")
            datasets.append(IMDBDataset)
            output_dimensions.append(2)
        elif dset == "AMAZON":
            dataset_names.append("AMAZON")
            datasets.append(AmazonDataset)
            output_dimensions.append(5)
        elif dset == "YAHOO":
            dataset_names.append("YAHOO")
            datasets.append(YahooDataset)
            output_dimensions.append(10)
    return output_dimensions, dataset_names, datasets