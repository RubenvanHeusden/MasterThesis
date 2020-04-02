import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data import Field
from torch.optim.lr_scheduler import StepLR
from codebase.experiments.multitask_classification.train_methods import *
from codebase.models.multitaskmodel import MultiTaskModel
from codebase.models.multitasklstm import MultiTaskLSTM
from codebase.models.mlp import MLP
from codebase.data_classes.data_utils import combine_datasets, multi_task_dataset_prep
import argparse
from codebase.data_classes.customdataloader import CustomDataLoader


def main(args):
    dataset_class, output_dimensions, target_names = multi_task_dataset_prep(args.dataset)
    # Set the random seed for experiments (check if I need to do this for all the other files as well)
    torch.cuda.empty_cache()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    include_lens = args.use_lengths
    TEXT = Field(lower=args.do_lowercase, include_lengths=args.use_lengths, batch_first=True,
                 fix_length=500)
    towers = {MLP(args.hidden_dim, args.linear_layers, output_dim): name for output_dim,
                                                                        name in zip(output_dimensions, target_names)}
    # Use name of dataset to get the arguments needed
    print("--- Starting with reading in the %s dataset ---" % args.dataset)
    dataset = dataset_class(text_field=TEXT).load(targets=target_names)[:-1]
    print("--- Finished with reading in the %s dataset ---" % args.dataset)
    # Load the dataset and split it into train and test portions
    dloader = CustomDataLoader(dataset, TEXT, target_names)
    data_iterators, total_vocab = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                                 batch_size=args.batch_size, device=torch.device("cpu"))



    model = MultiTaskLSTM(vocab=total_vocab,  hidden_dim=args.hidden_dim, device=args.device,
                            use_lengths=args.use_lengths)

    multitask_model = MultiTaskModel(shared_layer=model, towers=towers, batch_size=args.batch_size,
                                     input_dimension=args.embedding_dim, device=args.device,
                                     include_lens=args.use_lengths)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(multitask_model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma)

    train(multitask_model, criterion, optimizer, scheduler, list(data_iterators[0]), device=args.device,
          include_lengths=include_lens, save_path=args.logdir, save_name="%s_datasets" % "_".join(target_names),
          use_tensorboard=True, n_epochs=args.n_epochs, checkpoint_interval=args.save_interval,
          clip_val=args.gradient_clip)

    print("Evaluating model")
    multitask_model.load_state_dict(torch.load("saved_models/LSTM/%s_datasets_epoch_%d.pt" % ("_".join(target_names),
                                                                                              args.n_epochs-1)))
    for i, iterator in enumerate(data_iterators[0]):
        print("evaluating on dataset %s" % target_names[i])
        evaluation(multitask_model, iterator, criterion, device=args.device)


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
    parser.add_argument("--use_lengths", type=str, default="True")
    parser.add_argument("--do_lowercase", type=str, default="True")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=25)
    parser.add_argument("--linear_layers", nargs='+', required=True)
    parser.add_argument("--logdir", type=str, default="saved_models/LSTM")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--gradient_clip", type=float, default=0.0)
    args = parser.parse_args()

    args.use_lengths = eval(args.use_lengths)
    args.do_lowercase = eval(args.do_lowercase)
    args.linear_layers = list(map(int, args.linear_layers))
    main(args)
