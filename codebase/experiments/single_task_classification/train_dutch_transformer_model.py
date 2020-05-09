# External imports
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchtext.data import Field

# Local imports
from codebase.data_classes.data_utils import single_task_dataset_prep, single_task_class_weighting
from codebase.experiments.single_task_classification.train_methods import *
from codebase.data_classes.customdataloader import CustomDataLoader
from codebase.models.transformermodel import TransformerModel


def main(args):

    torch.cuda.empty_cache()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    # Lines below are make sure cuda is (almost) deterministic, can slow down training
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    TEXT = Field(lower=True, include_lengths=False, batch_first=True, fix_length=args.fix_length, init_token="[cls]")

    dataset_class, num_classes, target = single_task_dataset_prep(args.dataset)

    # Use name of dataset to get the arguments needed
    print("--- Starting with reading in the %s dataset ---" % args.dataset)
    dataset = dataset_class(text_field=TEXT, stratified_sampling=args.use_stratify).load(targets=target)
    # dataset.load returns train and test datasets
    print("--- Finished with reading in the %s dataset ---" % args.dataset)
    # Load the dataset and split it into train and test portions

    dloader = CustomDataLoader(dataset, TEXT, target)
    data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                                 batch_size=args.batch_size, device=torch.device("cpu"))

    words, embed_dict, embeddings, embed_dim = torch.load("../vector_cache/nl_320/combined-320.txt.pt")
    TEXT.vocab.set_vectors(embed_dict, embeddings, embed_dim)

    model = TransformerModel(max_seq_len=args.fix_length,
                             num_outputs=num_classes,
                             word_embedding_matrix=TEXT.vocab.vectors,
                             feed_fwd_dim=args.fwd_dim,
                             num_transformer_layers=args.num_transformer_layers,
                             num_transformer_heads=args.num_transformer_heads,
                             pos_encoding_dropout=args.pos_encoding_dropout,
                             classification_dropout=args.fc_layer_dropout,
                             batch_first=True,
                             pad_index=TEXT.vocab.stoi['pad'])

    if args.class_weighting:
        weights = single_task_class_weighting(data_iterators[0], num_classes)
        criterion = nn.CrossEntropyLoss(weight=weights.to(args.device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.90, 0.98), eps=10e-9)
    scheduler = StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma)

    train(model, criterion, optimizer, scheduler, data_iterators[0], device=args.device,
          include_lengths=False, save_path=args.logdir, save_name="%s_dataset" % args.dataset,
          tensorboard_dir=args.logdir+"/runs", n_epochs=args.n_epochs, checkpoint_interval=args.save_interval,
          clip_val=args.gradient_clip)

    print("Evaluating model")
    #model.load_state_dict(torch.load(args.logdir+"/%s_dataset_epoch_100.pt" % args.dataset))
    model.load_state_dict(torch.load(args.logdir+"/%s_dataset_epoch_%d.pt" % (args.dataset, args.n_epochs-1)))
    evaluation(model, data_iterators[-1], criterion, device=args.device, include_lengths=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset loading arguments
    parser.add_argument("--dataset", help="""string specifying the dataset to be used
                                        "options are:
                                        -   DAILYDIALOG-ACT
                                        -   DAILYDIALOG-EMOT
                                        -   DAILYDIALOG-TOPIC
                                        -   ENRON-EMOT
                                        -   ENRON-CAT
                                        """, type=str, default="ENRON-EMOT")

    parser.add_argument("--do_lowercase", type=str, default="True")
    parser.add_argument("--use_stratify", type=str, default="True")
    parser.add_argument("--class_weighting", type=str, default="False")
    parser.add_argument("--fix_length", type=int, default=None)

    # Training arguments
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1)
    parser.add_argument("--gradient_clip", type=float, default=0.0)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--scheduler_stepsize", type=float, default=0.1)

    # Data processing arguments
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Transformer specific arguments
    parser.add_argument("--num_transformer_heads", type=int, default=6)
    parser.add_argument("--num_transformer_layers", type=int, default=6)
    parser.add_argument("--fwd_dim", type=int, default=2048)
    parser.add_argument("--pos_encoding_dropout", type=float, default=0.1)
    parser.add_argument("--fc_layer_dropout", type=float, default=0.2)

    # Logging arguments
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="saved_models/Transformer")

    args = parser.parse_args()
    args.do_lowercase = eval(args.do_lowercase)
    args.use_stratify = eval(args.use_stratify)
    args.class_weighting = eval(args.class_weighting)
    main(args)
