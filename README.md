# MasterThesis

This repository contains both the (draft version) of the final thesis, as well as the codebase.


## Codebase

The code base contains the following folders:
  * Data classes
    - Contains the classes that represent datasets, these classes also handle file
      reading tokenization, stopword removal etc.
  * Experiments
    - Contains code for running the experiments, this is split in "single task learning" and "multitask learning"
  * Models
    - Contains various models used in the classification tasks (currently contains CNN, FeedForward Net, LSTM and Bidirectional LSTM)
  
## Installation

To run this package please follow the instructions below:

1. Download the package to your local computer
2. Unzip the folder in the desired location
3. cd into the root directory of the package 
4. run the following command: 
```
	pip install -r requirements.txt
```


### Word Embeddings

This project uses the 300D word embeddings from the Glove Project (). In case these
are already present on your system, these can be placed on the .vector_cache folder,
so that the directory tree looks like below: 
```
root
|--- codebase
|       |----experiments
|       |        |---- .vector_cache
|       |        |          |---- glove.6B.300d.txt

```
When the word embeddings are not found in this folder, they will be automatically downloaded 
and placed in this folder. 

### Datasets

Both the Enron Dataset(link) and the DailyDialog Dataset(link) are available online.
(add a script to download them automatically)

## Running experiments

The project contains two options for classification, namely single-task and multi-task 
classification. For the right format of the dataset for multi-task classification please
refer to 'File Format for Multitask Classification'. 

Both the single-task classification and the multi-task classification folders contain several
scripts of the form "train_XXX_model.py", these can be called from the command line with the 
arguments specified in the file to train and evaluate that specific model. 

For example, the below command trains an LSTM model on predicting the categories for the Enron dataset

```
python train_lstm_model.py --dataset ENRON-CAT --n_epochs 10 --fix_length 500 --tensorboard_dir runs/enron_categories_experiment1
```
### Selecting Datasets

For the Multitask Classification experiments all tasks within the dataset are 
automatically all selected and trained simultaneously. 

Below are the (currently) available datasets and the different task for each dataset

```
ENRON
	- ENRON-CAT
	- ENRON-EMOT
```


```
DAILYDIALOG
	- DAILYDIALOG-ACT
	- DAILYDIALOG-EMOT
	- DAILYDIALOG-TOPIC
```

More details on the datasets can be found in the Thesis document.

## Notes on using TensorBoard

By default, all the training scripts automatically log several statistics about 
the training process such as the average loss and accuracy per epoch, as well as 
the structure of the model used. By default these statistics are recorded in a
'runs' folder that is created at runtime. to view the statistics, start a terminal
and navigate to  "XXX_task_classification/saved_models/MODEL_NAME". then run the following command: 

```
tensorboard --logdir=runs 
```

Now, the Tensorboard application will launch and the link to navigate to to view
the statistics is shown.

To disable the logging, use "--tensorboard_dir None" when running the training scripts

