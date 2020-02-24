import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from codebase.models.multigatemixtureofexperts import MultiGateMixtureofExperts
from codebase.experiments.multitask_classification.train_methods import *
from codebase.models.convnet import ConvNet
from codebase.models.multitaskconvnet import MultitaskConvNet
from codebase.models.mlp import MLP
from codebase.data_classes.data_utils import combine_datasets, multi_task_dataset_prep


def main(args):

    torch.cuda.empty_cache()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    include_lens = args.use_lengths

    output_dimensions, dataset_names, datasets = multi_task_dataset_prep(args.datasets)

    towers = {MLP(args.num_filters_experts*len(args.filter_list_experts), args.linear_layers, output_dim): name for output_dim,
                                                                        name in zip(output_dimensions, dataset_names)}
    total_vocab, train_iterators, test_iterators = combine_datasets(datasets, include_lens=args.use_lengths,
                                                                    set_lowercase=args.do_lowercase,
                                                                    batch_size=args.batch_size, task_names=dataset_names)

    # initialize the multiple LSTMs and gating functions
    gating_networks = [ConvNet(input_channels=1, filter_list=args.filter_list_g,
                    embed_matrix=total_vocab, num_filters=args.num_filters_experts, output_dim=args.n_experts) for _ in
                                                range(len(dataset_names))]

    shared_layers = [MultitaskConvNet(input_channels=1, filter_list=args.filter_list_experts,
                    embed_matrix=total_vocab, num_filters=args.num_filters_experts)
                       for _ in range(args.n_experts)]

    model = MultiGateMixtureofExperts(shared_layers=shared_layers, gating_networks=gating_networks,
                                      towers=towers, device=args.device, include_lens=args.use_lengths,
                                      batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma)

    train(model, criterion, optimizer, scheduler, list(train_iterators), device=args.device,
          include_lengths=include_lens, save_path=args.logdir, save_name="%s_datasets" % "_".join(dataset_names),
          use_tensorboard=True, n_epochs=args.n_epochs, checkpoint_interval=args.save_interval,
          clip_val=args.gradient_clip)

    print("Evaluating model")
    model.load_state_dict(torch.load("saved_models/MoE/%s_datasets_epoch_%d.pt" % ("_".join(dataset_names),
                                                                                              args.n_epochs-1)))
    for i, iterator in enumerate(test_iterators):
        print("evaluating on dataset %s" % dataset_names[i])
        evaluation(model, iterator, criterion, device=args.device)


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
    parser.add_argument("--use_lengths", type=str, default="False")
    parser.add_argument("--do_lowercase", type=str, default="True")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim_experts", type=int, default=64)
    parser.add_argument("--hidden_dim_g", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=25)
    parser.add_argument("--linear_layers", nargs='+', required=True)
    parser.add_argument("--logdir", type=str, default="saved_models/LSTM")
    parser.add_argument("--n_experts", type=int, default=3)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--gradient_clip", type=float, default=0.0)
    parser.add_argument("--num_filters_g", type=int, default=2)
    parser.add_argument("--num_filters_experts", type=int, default=5)
    parser.add_argument("--filter_list_g", nargs='+', required=True)
    parser.add_argument("--filter_list_experts", nargs='+', required=True)
    args = parser.parse_args()

    args.use_lengths = eval(args.use_lengths)
    args.do_lowercase = eval(args.do_lowercase)
    args.linear_layers = list(map(int, args.linear_layers))
    args.filter_list_g = list(map(int, args.filter_list_g))
    args.filter_list_experts = list(map(int, args.filter_list_experts))

    main(args)



