import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data import Field, LabelField
from codebase.data_classes.imdbdataset import IMDBDataset
from codebase.data_classes.sttdataset import SSTDataset
from codebase.data_classes.customdataloader import CustomDataLoader
from codebase.data_classes.yelpdataset import YelpDataset
from codebase.models.simplelstm import SimpleLSTM
from torch.optim.lr_scheduler import StepLR
from codebase.experiments.single_task_classification.train_methods import *
import argparse

# TODO: clip gradients
# Set the random seed for experiments (check if I need to do this for all the other files as well)


def main(dataset_class, device, batch_size, random_seed, lr, scheduler_step_size, scheduler_gamma,
         use_lengths, do_lowercase, embedding_dim, output_dim, hidden_dim, n_epochs, logdir,
         dataset_name=None):

    torch.cuda.empty_cache()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # Lines below are make sure cuda is (almost) deterministic, can slow down training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    include_lens = use_lengths
    TEXT = Field(lower=do_lowercase, include_lengths=include_lens, batch_first=True)
    # TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=True, batch_first=True)
    LABEL = LabelField(dtype=torch.long)
    print("--- Starting with reading in the %s dataset ---" % dataset_name)
    #dataset = SSTDataset(TEXT, LABEL, path="../.data/imdb/aclImdb").load()
    dataset = dataset_class(TEXT, LABEL).load()
    #dataset = YelpDataset(TEXT, LABEL, path="../.data/yelp").load()
    print("--- Finished with reading in the %s dataset ---" % dataset_name)
    # Load the dataset and split it into train and test portions
    dloader = CustomDataLoader(dataset, TEXT, LABEL)
    data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                                 batch_size=batch_size, device=device)


    # Some sample models that can be used are listed below, uncomment the particular model to use it

    model = SimpleLSTM(vocab=TEXT.vocab.vectors, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                  device=device, use_lengths=include_lens)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    train(model, criterion, optimizer, scheduler, data_iterators[0], device=device, include_lengths=include_lens,
        save_path=logdir, save_name="%s_dataset" % dataset_name, use_tensorboard=False, n_epochs=n_epochs)

    print("Evaluating model")
    model.load_state_dict(torch.load(logdir+"/%s_dataset_epoch_%d.pt" % (dataset_name, n_epochs-1)))
    evaluation(model, data_iterators[-1], criterion, device=device, include_lengths=include_lens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="""string specifying the dataset to be used
                                        "options are:
                                        -   SST
                                        -   YELP
                                        -   IMDB
                                        """, type=str, default="SST")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1)
    parser.add_argument("--scheduler_stepsize", type=float, default=0.1)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--use_lengths", type=str, default="True")
    parser.add_argument("--do_lowercase", type=str, default="True")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=25)
    # TODO: set the right path here
    parser.add_argument("--logdir", type=str, default="saved_models/LSTM")
    args = parser.parse_args()
    args.use_lengths = eval(args.use_lengths)
    args.do_lowercase = eval(args.do_lowercase)
    if args.dataset == "SST":
        dataset = SSTDataset
        output_dim = 2
    elif args.dataset == "YELP":
        dataset = YelpDataset
        output_dim = 5
    elif args.dataset == "IMDB":
        dataset = IMDBDataset
        output_dim = 2
    else:
        raise(Exception("Invalid dataset argument, please refer to the help function of the "
                        "argument parser for details on valid arguments"))

    model = SimpleLSTM
    # run the main program
    main(dataset, args.device, args.batch_size, args.random_seed, args.learning_rate,
         args.scheduler_stepsize, args.scheduler_gamma,
         args.use_lengths, args.do_lowercase, args.embedding_dim, output_dim, args.hidden_dim,
         args.n_epochs, args.logdir, dataset_name=args.dataset)
