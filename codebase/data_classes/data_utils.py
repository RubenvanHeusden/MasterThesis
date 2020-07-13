import torch
import pandas as pd
from torchtext.data import Example, Dataset
from codebase.data_classes.csvdataset import CSVDataset
from sklearn.utils.class_weight import compute_class_weight
from codebase.data_classes.bertemeddingenrondataset import BertEmbeddingEnronDataset
from codebase.data_classes.bertembeddingcustomdataset import BertEmbeddingCustomDataset
from codebase.data_classes.bertembeddingdailydialogdataset import BertEmbeddingDailyDialogDataset


def get_num_classes_dataset(path_to_dataset, target_names, delimiter: str = ",", quotechar='"'):
    num_classes = []

    dataset_train = pd.read_csv(path_to_dataset+"train.csv", sep=delimiter, quotechar=quotechar)
    dataset_test = pd.read_csv(path_to_dataset+"test.csv", sep=delimiter, quotechar=quotechar)
    dataset = pd.concat((dataset_train, dataset_test), axis=0)

    for target in target_names:
        num_classes.append(dataset[target].nunique())
    if len(num_classes) == 1:
        return num_classes[0]
    return num_classes


def single_task_class_weighting(dataset):
    total_y = []
    for X, y, _ in dataset:
        total_y.append(y)

    total_y = torch.cat(total_y, dim=0)
    classes = torch.unique(total_y)
    weights = compute_class_weight('balanced', classes=classes.data.numpy(),
                                   y=total_y.data.numpy())
    return torch.from_numpy(weights).float()


def multitask_class_weighting(dataset, target_names):

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


