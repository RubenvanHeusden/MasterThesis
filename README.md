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

If you want to use your own dataset, you have to make sure that the first two columns are for the ID of the 
datapoint and the text of the datapoint, like shown below: 

<table style="width:100%">
  <tr>
    <th>ID</th>
    <th>text</th> 
    <th>task_a_labels</th>
    <th> task_b_labels</th>
  </tr>
  <tr>
    <td>1</td>
    <td>blabla</td>
    <td>class_a</td>
    <td>0</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Hello World!</td>
    <td>class_b</td>
    <td>1</td>
  </tr>
  <tr>
    <td>3</td>
    <td>some more text</td>
    <td>class_b</td>
    <td>0</td>
  </tr>
</table>
 
 when running the training commands, set ``--data_path`` to the folder that 
 contains the  train.csv and test.csv. now for the ``--target_names`` set the names of the 
 task labels. In this case this would be ``--target_names task_a_labels task_b_labels``

An example command for running the multigate Mixture-of-Experts model with CNN gating networks:
```python train_multigate_mixture_of_experts_cnn.py  --data_path ../.data/enron/ --target_names category emotion --learning_rate 0.1 --n_epochs 50 --save_interval 500 --batch_size 64 --fix_length 100 --class_weighting True --linear_layers 256 --random_seed 3216 --num_filters_experts 100 --filter_list_experts 3 4 5 --num_filters_g 25 --filter_list_g 3 4 5 --n_experts 3 --balancing_strategy "dropout" --balance_epoch_cnt 100 --gate_dropout 0.30 --gating_nets_type CNN```

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

