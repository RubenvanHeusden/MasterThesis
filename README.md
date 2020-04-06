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
(fill in later)

This project uses the 300D word embeddings from the Glove Project (). In case these
are already present on your system, these can be placed on the .vector_cache folder,
so that the directory tree looks like below: 
```


```
When the word embeddings are not found in this folder, they will be automatically downloaded 
and placed in this folder. 

### Datasets



## Running experiments

The project contains two options for classification, namely single-task and multi-task 
classification. For the right format of the dataset for multi-task classification please
refer to 'File Format for Multitask Classification'. 


### File Format for Multitask Classification







