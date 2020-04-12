import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data import Field
from torch.optim.lr_scheduler import StepLR
from codebase.experiments.multitask_classification.train_methods import *
from codebase.models.multitaskmodel import MultiTaskModel
from codebase.models.multitasktransformer import MultiTaskTransformerModel
from codebase.models.mlp import MLP
from codebase.data_classes.data_utils import multi_task_dataset_prep, multitask_class_weighting
import argparse
from codebase.data_classes.customdataloader import CustomDataLoader


def main(args):
    dataset_class, output_dimensions, target_names = multi_task_dataset_prep(args.dataset)
    torch.cuda.empty_cache()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=False,
                 batch_first=True, fix_length=args.fix_length, init_token="[cls]")
    towers = {MLP(300, args.linear_layers, output_dim): name for output_dim,
                                                                        name in zip(output_dimensions, target_names)}
    # Use name of dataset to get the arguments needed
    print("--- Starting with reading in the %s dataset ---" % args.dataset)
    dataset = dataset_class(text_field=TEXT).load(targets=target_names)
    print("--- Finished with reading in the %s dataset ---" % args.dataset)
    # Load the dataset and split it into train and test portions

    dloader = CustomDataLoader(dataset, TEXT, target_names)
    data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                                 batch_size=args.batch_size, device=torch.device("cpu"))

    model = MultiTaskTransformerModel(max_seq_len=args.fix_length,
                             word_embedding_matrix=TEXT.vocab.vectors,
                             feed_fwd_dim=args.fwd_dim,
                             num_transformer_layers=args.num_transformer_layers,
                             num_transformer_heads=args.num_transformer_heads,
                             pos_encoding_dropout=args.pos_encoding_dropout,
                             classification_dropout=args.fc_layer_dropout,
                             batch_first=True,
                             pad_index=TEXT.vocab.stoi['pad'])

    multitask_model = MultiTaskModel(shared_layer=model, towers=towers, batch_size=args.batch_size,
                                     input_dimension=TEXT.vocab.vectors.shape[1], device=args.device,
                                     include_lens=False)

    if args.class_weighting:
        task_weights = multitask_class_weighting(data_iterators[0], target_names, output_dimensions)
        losses = {name: nn.CrossEntropyLoss(weight=task_weights[name].to(args.device)) for name in target_names}
    else:
        losses = {name: nn.CrossEntropyLoss() for name in target_names}

    optimizer = optim.SGD(multitask_model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma)

    train(multitask_model, losses, optimizer, scheduler, data_iterators[0], device=args.device,
          include_lengths=False, save_path=args.logdir, save_name="%s_datasets" % "_".join(target_names),
          tensorboard_dir=args.logdir+"/runs", n_epochs=args.n_epochs, checkpoint_interval=args.save_interval,
          clip_val=args.gradient_clip)

    print("Evaluating model")
    multitask_model.load_state_dict(torch.load("%s/%s_datasets_epoch_%d.pt" % (args.logdir, "_".join(target_names),
                                                                                              args.n_epochs-1)))
    evaluation(multitask_model, data_iterators[-1], losses, device=args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset loading arguments
    parser.add_argument("--dataset", help="""string specifying the dataset to be used
                                        "options are:
                                            - ENRON
                                            - DAILYDIALOG
                                        """, type=str, default="ENRON")

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Transformer specific arguments
    parser.add_argument("--num_transformer_heads", type=int, default=1)
    parser.add_argument("--num_transformer_layers", type=int, default=1)
    parser.add_argument("--fwd_dim", type=int, default=8)
    parser.add_argument("--pos_encoding_dropout", type=float, default=0.1)
    parser.add_argument("--fc_layer_dropout", type=float, default=0.2)
    parser.add_argument("--linear_layers", nargs='+', required=True)
    # Logging arguments
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="saved_models/Transformer")

    args = parser.parse_args()
    args.do_lowercase = eval(args.do_lowercase)
    args.use_stratify = eval(args.use_stratify)
    args.class_weighting = eval(args.class_weighting)
    args.linear_layers = list(map(int, args.linear_layers))
    main(args)
