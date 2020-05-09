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
from sklearn.utils.class_weight import compute_class_weight


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

    elif dataset_string == "CUSTOM-CAT":
        dataset = CustomDataset
        output_dim = 18
        target = ('label', )

    elif dataset_string == "CUSTOM-EMOT":
        dataset = CustomDataset
        output_dim = 8
        target = ("emotion_classes", )

    elif dataset_string == "CUSTOM-INTENT":
        dataset = CustomDataset
        output_dim = 5
        target = ("intent_classes", )

    else:
        raise(Exception("Invalid dataset argument, please refer to the help function of the "
                        "argument parser for details on valid arguments"))
    return dataset, output_dim, target


def multi_task_dataset_prep(dataset_string):
    if dataset_string == "DAILYDIALOG":
        dataset = DailyDialogDataset
        output_dim = [7, 4, 10]
        targets = ("emotion", "act", "topic")

    elif dataset_string == "ENRON":
        dataset = EnronDataset
        output_dim = [6, 10]
        targets = ("category", "emotion")

    elif dataset_string == "CUSTOM":
        dataset = CustomDataset
        output_dim = [18, 5, 8]
        targets = ("label", "intent_classes", "emotion_classes")

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


def single_task_class_weighting(dataset, num_classes):
    total_y = []
    for X, y, _ in dataset:
        total_y.append(y)

    total_y = torch.cat(total_y, dim=0)
    classes = torch.unique(total_y)
    weights = compute_class_weight('balanced', classes=classes.data.numpy(),
                                   y=total_y.data.numpy())
    return torch.from_numpy(weights).float()


def multitask_class_weighting(dataset, target_names, num_classes):

    task_weights = {}
    task_totals = {task: [] for task in target_names}

    for X, *targets, tasks in dataset:
        for task, task_name in zip(targets, tasks):
            task_totals[task_name].append(task)
    for name in target_names:
        task_y = torch.cat(task_totals[name], dim=0)
        unique_task = torch.unique(task_y)
        task_weights[name] = torch.from_numpy(compute_class_weight('balanced', classes=unique_task.data.numpy(),
                                                  y=task_y.data.numpy())).float()

    return task_weights
