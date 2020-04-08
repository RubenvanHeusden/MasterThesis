import torch
import itertools
from typing import List, Any
from torchtext.data import Field, LabelField, Example, TabularDataset, Dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedShuffleSplit
from codebase.data_classes.dailydialogdataset import DailyDialogDataset
from codebase.data_classes.enrondataset import EnronDataset
from codebase.data_classes.customdataset import CustomDataset
import numpy as np


def single_task_dataset_prep(dataset_string):
    if dataset_string == "DAILYDIALOG-EMOT":
        dataset = DailyDialogDataset
        output_dim = 7
        target = ("emotion", )

    elif dataset_string == "DAILYDIALOG-TOPIC":
        dataset = DailyDialogDataset
        output_dim = 10
        target = ("topic", )

    elif dataset_string == "DAILYDIALOG-ACT":
        dataset = DailyDialogDataset
        output_dim = 4
        target = ("act", )

    elif dataset_string == "ENRON-CAT":
        dataset = EnronDataset
        output_dim = 6
        target = ("category", )

    elif dataset_string == "ENRON-EMOT":
        dataset = EnronDataset
        output_dim = 10
        target = ("emotion", )

    elif dataset_string == "CUSTOM":
        dataset = CustomDataset
        output_dim = 17
        target = ('label', )
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
        output_dim = [6, 10]
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

