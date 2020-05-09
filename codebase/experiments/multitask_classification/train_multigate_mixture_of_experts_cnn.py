import argparse
from torchtext.data import Field
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from codebase.models.multigatemixtureofexperts import MultiGateMixtureofExperts
from codebase.experiments.multitask_classification.train_methods import *
from codebase.models.convnet import ConvNet
from codebase.models.multitaskconvnet import MultitaskConvNet
from codebase.models.mlp import MLP
from codebase.data_classes.data_utils import multi_task_dataset_prep, multitask_class_weighting
from codebase.data_classes.customdataloader import CustomDataLoader


def main(args):
    dataset_class, output_dimensions, target_names = multi_task_dataset_prep(args.dataset)
    torch.cuda.empty_cache()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=args.use_lengths, batch_first=True,
                 fix_length=args.fix_length)

    towers = {MLP(args.num_filters_experts*len(args.filter_list_experts), args.linear_layers, output_dim): name for output_dim,
                                                                        name in zip(output_dimensions, target_names)}

    # Use name of dataset to get the arguments needed
    print("--- Starting with reading in the %s dataset ---" % args.dataset)
    dataset = dataset_class(text_field=TEXT).load(targets=target_names)
    print("--- Finished with reading in the %s dataset ---" % args.dataset)
    # Load the dataset and split it into train and test portions

    dloader = CustomDataLoader(dataset, TEXT, target_names)
    data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                                 batch_size=args.batch_size, device=torch.device("cpu"))

    # initialize the multiple LSTMs and gating functions
    gating_networks = [ConvNet(input_channels=1, filter_list=args.filter_list_g,
                    embed_matrix=TEXT.vocab.vectors, num_filters=args.num_filters_experts, output_dim=args.n_experts) for _ in
                                                range(len(target_names))]

    shared_layers = [MultitaskConvNet(input_channels=1, filter_list=args.filter_list_experts,
                    embed_matrix=TEXT.vocab.vectors, num_filters=args.num_filters_experts)
                       for _ in range(args.n_experts)]

    model = MultiGateMixtureofExperts(shared_layers=shared_layers, gating_networks=gating_networks,
                                      towers=towers, device=args.device, include_lens=args.use_lengths,
                                      batch_size=args.batch_size, gating_drop=args.gate_dropout)

    if args.class_weighting:
        task_weights = multitask_class_weighting(data_iterators[0], target_names, output_dimensions)
        losses = {name: nn.CrossEntropyLoss(weight=task_weights[name].to(args.device)) for name in target_names}
    else:
        losses = {name: nn.CrossEntropyLoss() for name in target_names}

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma)

    train(model, losses, optimizer, scheduler, data_iterators[0], device=args.device,
          include_lengths=args.use_lengths, save_path=args.logdir, save_name="%s_datasets" % "_".join(target_names),
          tensorboard_dir=args.logdir+"/runs", n_epochs=args.n_epochs, checkpoint_interval=args.save_interval,
          clip_val=args.gradient_clip)

    print("Evaluating model")
    model.load_state_dict(torch.load("%s/%s_datasets_epoch_%d.pt" % (args.logdir, "_".join(target_names),
                                                                                              args.n_epochs-1)))
    evaluation(model, data_iterators[-1], losses, device=args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="""string specifying the dataset to be used
                                        "options are:
                                        -   DAILYDIALOG
                                        -   ENRON
                                        """, type=str, default="ENRON")

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
    parser.add_argument("--fix_length", type=int, default=None)
    parser.add_argument("--class_weighting", type=str, default="False")
    parser.add_argument("--gate_dropout", type=float, default=0.0)
    args = parser.parse_args()

    args.use_lengths = eval(args.use_lengths)
    args.do_lowercase = eval(args.do_lowercase)
    args.linear_layers = list(map(int, args.linear_layers))
    args.filter_list_g = list(map(int, args.filter_list_g))
    args.filter_list_experts = list(map(int, args.filter_list_experts))

    main(args)



