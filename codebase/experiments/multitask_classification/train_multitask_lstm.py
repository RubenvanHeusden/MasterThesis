import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data import Field
from torch.optim.lr_scheduler import StepLR
from codebase.experiments.multitask_classification.train_methods import *
from codebase.models.multitaskmodel import MultiTaskModel
from codebase.models.multitasklstm import MultiTaskLSTM
from codebase.models.mlp import MLP
from codebase.data_classes.data_utils import get_num_classes_dataset, multitask_class_weighting
import argparse
from codebase.data_classes.customdataloader import CustomDataLoader
from codebase.data_classes.csvdataset import CSVDataset


def main(args):

    torch.cuda.empty_cache()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=args.use_lengths,
                 batch_first=True, fix_length=args.fix_length)

    output_dimensions = get_num_classes_dataset(args.data_path, args.target_names)

    # Use name of dataset to get the arguments needed
    print("--- Starting with reading in the dataset ---")
    dataset = CSVDataset(text_field=TEXT, path_to_datadir=args.data_path).load(targets=args.target_names)
    print("--- Finished with reading in the dataset ---")

    towers = {MLP(args.hidden_dim, args.linear_layers, output_dim): name for output_dim,
                                                                        name in zip(output_dimensions, args.target_names)}
    # Load the dataset and split it into train and test portions

    dloader = CustomDataLoader(dataset, TEXT, args.target_names)
    data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                                 batch_size=args.batch_size, device=torch.device("cpu"))

    model = MultiTaskLSTM(vocab=TEXT.vocab.vectors,  hidden_dim=args.hidden_dim, device=args.device,
                            use_lengths=args.use_lengths)

    multitask_model = MultiTaskModel(shared_layer=model, towers=towers, batch_size=args.batch_size,
                                     input_dimension=args.embedding_dim, device=args.device,
                                     include_lens=args.use_lengths)

    if args.class_weighting:
        task_weights = multitask_class_weighting(data_iterators[0], args.target_names)
        losses = {name: nn.CrossEntropyLoss(weight=task_weights[name].to(args.device)) for name in args.target_names}
    else:
        losses = {name: nn.CrossEntropyLoss() for name in args.target_names}

    optimizer = optim.SGD(multitask_model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma)

    train(multitask_model, losses, optimizer, scheduler, data_iterators[0], device=args.device,
          include_lengths=args.use_lengths, save_path=args.logdir, save_name="%s_datasets" % "_".join(args.target_names),
          tensorboard_dir=args.logdir+"/runs", n_epochs=args.n_epochs, checkpoint_interval=args.save_interval,
          clip_val=args.gradient_clip)

    print("Evaluating model")
    multitask_model.load_state_dict(torch.load("%s/%s_datasets_epoch_%d.pt" % (args.logdir, "_".join(args.target_names),
                                                                                              args.n_epochs-1)))
    evaluation(multitask_model, data_iterators[-1], losses, device=args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_names", nargs="+", required=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1)
    parser.add_argument("--scheduler_stepsize", type=float, default=0.1)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--gradient_clip", type=float, default=0.0)

    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=25)

    parser.add_argument("--use_lengths", type=str, default="True")
    parser.add_argument("--do_lowercase", type=str, default="True")
    parser.add_argument("--fix_length", type=int, default=None)
    parser.add_argument("--class_weighting", type=str, default="False")

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--linear_layers", nargs='+', required=True)

    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--logdir", type=str, default="saved_models/Multitask_LSTM")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    args.use_lengths = eval(args.use_lengths)
    args.do_lowercase = eval(args.do_lowercase)
    args.linear_layers = list(map(int, args.linear_layers))
    args.target_names = tuple(list(map(str, args.target_names)))
    main(args)
