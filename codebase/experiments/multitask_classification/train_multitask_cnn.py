import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field
import numpy as np
from torch.optim.lr_scheduler import StepLR
from codebase.experiments.multitask_classification.train_methods import *
from codebase.models.multitaskmodel import MultiTaskModel
from codebase.models.multitaskconvnet import MultitaskConvNet
from codebase.data_classes.customdataloader import CustomDataLoader
from codebase.models.mlp import MLP
from codebase.data_classes.data_utils import multi_task_dataset_prep, multitask_class_weighting
import argparse


def main(args):

    # TODO: clip gradients
    # for the multitask learning, make a dictionary containing "task": data
    dataset_class, output_dimensions, target_names = multi_task_dataset_prep(args.dataset)
    # Set the random seed for experiments (check if I need to do this for all the other files as well)
    torch.cuda.empty_cache()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=args.use_lengths, batch_first=True,
                 fix_length=args.fix_length)
    # Load datasets

    # Use name of dataset to get the arguments needed
    print("--- Starting with reading in the %s dataset ---" % args.dataset)
    dataset = dataset_class(text_field=TEXT).load(targets=target_names)
    print("--- Finished with reading in the %s dataset ---" % args.dataset)
    # Load the dataset and split it into train and test portions

    dloader = CustomDataLoader(dataset, TEXT, target_names)
    data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                                 batch_size=args.batch_size, device=torch.device("cpu"))

    towers = {MLP(len(args.filter_list)*args.num_filters, args.linear_layers, output_dim): name for output_dim,
                                                                        name in zip(output_dimensions, target_names)}

    model = MultitaskConvNet(1, args.filter_list, TEXT.vocab.vectors, args.num_filters, dropbout_probs=args.dropout)

    multitask_model = MultiTaskModel(shared_layer=model, towers=towers, batch_size=args.batch_size,
                                     input_dimension=TEXT.vocab.vectors.shape[1], device=args.device,
                                     include_lens=args.use_lengths)

    if args.class_weighting:
        task_weights = multitask_class_weighting(data_iterators[0], target_names, output_dimensions)
        losses = {name: nn.CrossEntropyLoss(weight=task_weights[name].to(args.device)) for name in target_names}
    else:
        losses = {name: nn.CrossEntropyLoss() for name in target_names}

    optimizer = optim.SGD(multitask_model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma)

    train(multitask_model, losses, optimizer, scheduler, data_iterators[0], device=args.device,
          include_lengths=args.use_lengths, save_path=args.logdir, save_name="%s_datasets" % "_".join(target_names),
          tensorboard_dir=args.logdir+"/runs", n_epochs=args.n_epochs, checkpoint_interval=args.save_interval,
          clip_val=args.gradient_clip)

    print("Evaluating model")
    multitask_model.load_state_dict(torch.load("%s/%s_datasets_epoch_%d.pt" % (args.logdir, "_".join(target_names),
                                                                                              args.n_epochs-1)))
    evaluation(multitask_model, data_iterators[-1], losses, device=args.device)


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
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--n_epochs", type=int, default=25)
    parser.add_argument("--logdir", type=str, default="saved_models/CNN")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--gradient_clip", type=float, default=0.0)

    parser.add_argument("--use_lengths", type=str, default="False")
    parser.add_argument("--do_lowercase", type=str, default="True")
    parser.add_argument("--class_weighting", type=str, default="False")
    parser.add_argument("--fix_length", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--num_filters", type=int, default=100)
    parser.add_argument("--filter_list", nargs='+', required=True)
    parser.add_argument("--linear_layers", nargs='+', required=True)

    args = parser.parse_args()

    args.use_lengths = eval(args.use_lengths)
    args.do_lowercase = eval(args.do_lowercase)
    args.filter_list = list(map(int, args.filter_list))
    args.linear_layers = list(map(int, args.linear_layers))
    main(args)
