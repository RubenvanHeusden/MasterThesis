# External imports
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchtext.data import Field, LabelField

# Local imports
from codebase.models.convnet import ConvNet
from codebase.models.simplemoe import SimpleMoE
from codebase.data_classes.customdataloader import CustomDataLoader
from codebase.data_classes.data_utils import single_task_dataset_prep
from codebase.experiments.single_task_classification.train_methods import *


def main(args):

    torch.cuda.empty_cache()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    # Lines below are make sure cuda is (almost) deterministic, can slow down training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    TEXT = Field(lower=args.do_lowercase, include_lengths=args.use_lengths, batch_first=True)
    # TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=True, batch_first=True)
    LABEL = LabelField(dtype=torch.long)
    dataset_class, num_classes = single_task_dataset_prep(args.dataset)
    print("--- Starting with reading in the %s dataset ---" % args.dataset)
    # dataset = SSTDataset(TEXT, LABEL, path="../.data/imdb/aclImdb").load()
    dataset = dataset_class(TEXT, LABEL).load()
    # dataset = YelpDataset(TEXT, LABEL, path="../.data/yelp").load()
    print("--- Finished with reading in the %s dataset ---" % args.dataset)
    # Load the dataset and split it into train and test portions
    dloader = CustomDataLoader(dataset, TEXT, LABEL)
    data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                                 batch_size=args.batch_size, device=torch.device("cpu"))

    g = ConvNet(input_channels=1, output_dim=args.n_experts, filter_list=args.filter_list_g,
                    embed_matrix=TEXT.vocab.vectors, num_filters=args.num_filters_g)

    expert_networks = [ConvNet(input_channels=1, output_dim=num_classes, filter_list=args.filter_list_experts,
                    embed_matrix=TEXT.vocab.vectors, num_filters=args.num_filters_experts)
                       for _ in range(args.n_experts)]

    model = SimpleMoE(gating_network=g, expert_networks=
    expert_networks, output_dim=num_classes, device=args.device, include_lens=args.use_lengths)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma)

    train(model, criterion, optimizer, scheduler, data_iterators[0], device=args.device, include_lengths=args.use_lengths,
        save_path=args.logdir, save_name="%s_dataset" % args.dataset, use_tensorboard=False, n_epochs=args.n_epochs,
              checkpoint_interval=args.save_interval, clip_val=args.gradient_clip)

    print("Evaluating model")
    model.load_state_dict(torch.load(args.logdir+"/%s_dataset_epoch_%d.pt" % (args.dataset, args.n_epochs-1)))
    evaluation(model, data_iterators[-1], criterion, device=args.device, include_lengths=args.use_lengths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset loading arguments
    parser.add_argument("--dataset", help="""string specifying the dataset to be used
                                        "options are:
                                        -   SST
                                        -   YELP
                                        -   IMDB
                                        """, type=str, default="SST")
    parser.add_argument("--use_lengths", type=str, default="False")
    parser.add_argument("--do_lowercase", type=str, default="True")

    # Training arguments
    parser.add_argument("--n_epochs", type=int, default=25)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1)
    parser.add_argument("--gradient_clip", type=float, default=0.0)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--scheduler_stepsize", type=float, default=0.1)

    # Data processing arguments
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # MoE-CNN model arguments
    parser.add_argument("--n_experts", type=int, default=3)
    parser.add_argument("--num_filters_g", type=int, default=2)
    parser.add_argument("--filter_list_g", nargs='+', required=True)
    parser.add_argument("--num_filters_experts", type=int, default=5)
    parser.add_argument("--filter_list_experts", nargs='+', required=True)

    # Logging arguments
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--logdir", type=str, default="saved_models/MoE_CNN")

    args = parser.parse_args()

    args.use_lengths = eval(args.use_lengths)
    args.do_lowercase = eval(args.do_lowercase)
    args.filter_list_g = list(map(int, args.filter_list_g))
    args.filter_list_experts = list(map(int, args.filter_list_experts))

    main(args)

