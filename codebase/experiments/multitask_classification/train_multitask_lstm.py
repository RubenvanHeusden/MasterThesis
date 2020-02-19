import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from codebase.experiments.multitask_classification.train_methods import *
from codebase.models.multitaskmodel import MultiTaskModel
from codebase.models.multitasklstm import MultiTaskLSTM
from codebase.models.mlp import MLP
from codebase.data_classes.data_utils import combine_datasets, multi_task_dataset_prep
import argparse


def main(dataset_classes, device, batch_size, random_seed, lr, scheduler_step_size, scheduler_gamma,
         use_lengths, do_lowercase, embedding_dim, output_dims, hidden_dim, linear_layers_towrs, n_epochs, logdir,
         dataset_names=None):

    # TODO: clip gradients
    # for the multitask learning, make a dictionary containing "task": data

    # Set the random seed for experiments (check if I need to do this for all the other files as well)
    torch.cuda.empty_cache()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    include_lens = use_lengths

    towers = {MLP(hidden_dim, linear_layers_towrs, output_dim): name for output_dim,
                                                                        name in zip(output_dims, dataset_names)}
    total_vocab, train_iterators, test_iterators = combine_datasets(dataset_classes, include_lens=use_lengths,
                                                                    set_lowercase=do_lowercase, batch_size=batch_size,
                           task_names=dataset_names, device=device)

    model = MultiTaskLSTM(vocab=total_vocab, embedding_dim=embedding_dim, hidden_dim=hidden_dim, device=device,
                            use_lengths=use_lengths)

    multitask_model = MultiTaskModel(shared_layer=model, towers=towers, batch_size=batch_size,
                                     input_dimension=embedding_dim, device=device, include_lens=use_lengths)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(multitask_model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    train(multitask_model, criterion, optimizer, scheduler, list(train_iterators), device=device,
          include_lengths=include_lens, save_path=logdir, save_name="%s_datasets" % "_".join(dataset_names),
          use_tensorboard=True, n_epochs=n_epochs)

    print("Evaluating model")
    multitask_model.load_state_dict(torch.load("saved_models/LSTM/%s_datasets_epoch_%d.pt" % ("_".join(dataset_names),
                                                                                              n_epochs-1)))
    for i, iterator in enumerate(test_iterators):
        print("evaluating on dataset %s" % dataset_names[i])
        evaluation(multitask_model, iterator, criterion, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", help="""string specifying the dataset to be used
                                        "options are:
                                        -   SST
                                        -   YELP
                                        -   IMDB
                                        """, nargs='+', required=True)

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
    parser.add_argument("--linear_layers", nargs='+', required=True)
    parser.add_argument("--logdir", type=str, default="saved_models/LSTM")
    args = parser.parse_args()

    args.use_lengths = eval(args.use_lengths)
    args.do_lowercase = eval(args.do_lowercase)
    lin_layers = list(map(int, args.linear_layers))
    output_dimensions, dataset_names, datasets = multi_task_dataset_prep(args.datasets)
    main(datasets, args.device, args.batch_size, args.random_seed, args.learning_rate, args.scheduler_stepsize,
         args.scheduler_gamma, args.use_lengths, args.do_lowercase, args.embedding_dim, output_dimensions,
         args.hidden_dim, lin_layers, args.n_epochs, args.logdir, dataset_names=dataset_names)
