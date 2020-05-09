import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data import Field
from torch.optim.lr_scheduler import StepLR
from codebase.experiments.multitask_classification.train_methods import *
from codebase.models.multitaskmodel import MultiTaskModel
from codebase.models.multitasklstm import MultiTaskLSTM
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

    TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=args.use_lengths, batch_first=True,
                 fix_length=args.fix_length)
    towers = {MLP(args.hidden_dim, args.linear_layers, output_dim): name for output_dim,
                                                                        name in zip(output_dimensions, target_names)}
    # Use name of dataset to get the arguments needed
    print("--- Starting with reading in the %s dataset ---" % args.dataset)
    dataset = dataset_class(text_field=TEXT).load(targets=target_names)
    print("--- Finished with reading in the %s dataset ---" % args.dataset)
    # Load the dataset and split it into train and test portions

    dloader = CustomDataLoader(dataset, TEXT, target_names)
    data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                                 batch_size=args.batch_size, device=torch.device("cpu"))

    words, embed_dict, embeddings, embed_dim = torch.load("../vector_cache/nl_300/cc.nl.300.vec.pt")
    TEXT.vocab.set_vectors(embed_dict, embeddings, embed_dim)

    model = MultiTaskLSTM(vocab=TEXT.vocab.vectors,  hidden_dim=args.hidden_dim, device=args.device,
                            use_lengths=args.use_lengths)

    multitask_model = MultiTaskModel(shared_layer=model, towers=towers, batch_size=args.batch_size,
                                     input_dimension=args.embedding_dim, device=args.device,
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

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1)
    parser.add_argument("--scheduler_stepsize", type=float, default=0.1)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--use_lengths", type=str, default="True")
    parser.add_argument("--do_lowercase", type=str, default="True")
    parser.add_argument("--fix_length", type=int, default=None)
    parser.add_argument("--class_weighting", type=str, default="False")

    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=256)
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
