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
from codebase.models.simplelstm import SimpleLSTM
from codebase.data_classes.data_utils import multitask_class_weighting, get_num_classes_dataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


def collate_fn_enron(batch, include_lens=False):
    result = default_collate(batch)
    if include_lens:
        return (result[0], result[1]), result[2], result[3], batch[-1][-1]
    else:
        return result[0], result[2], result[3], batch[-1][-1]


def collate_fn_dailydialog(batch, include_lens=False):
    result = default_collate(batch)
    if include_lens:
        return (result[0], result[1]), result[2], result[3], result[4], batch[-1][-1]
    else:
        return result[0], result[2], result[3], result[4], batch[-1][-1]


def main(args):
    print("--- Starting with reading in the %s dataset ---" % args.dataset)
    (train_set, test_set), output_dimensions, target_names = multi_task_dataset_prep(args.dataset)
    print("--- Finished with reading in the %s dataset ---" % args.dataset)

    if args.dataset == "DAILYDIALOG-BERT":
        collate_fn = collate_fn_dailydialog
    elif args.dataset == "ENRON-BERT":
        collate_fn = collate_fn_enron
    else:
        raise (Exception("The given dataset name is not recognised"))

    train_set = DataLoader(train_set, shuffle=True, batch_size=args.batch_size,
                           collate_fn=lambda x: collate_fn(x, include_lens=args.use_lengths))
    test_set = DataLoader(test_set, shuffle=False, batch_size=args.batch_size,
                          collate_fn=lambda x: collate_fn(x, include_lens=args.use_lengths))

    torch.cuda.empty_cache()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    towers = {MLP(args.num_filters_experts*len(args.filter_list_experts), args.linear_layers, output_dim): name for output_dim,
                                                                        name in zip(output_dimensions, target_names)}

    # initialize the multiple LSTMs and gating functions
    # gating_networks = [ConvNet(input_channels=1, filter_list=args.filter_list_g,
    #                 embed_matrix=torch.zeros(size=(1, 768)), num_filters=args.num_filters_g, output_dim=args.n_experts,
    #                            use_bert_embeds=True) for _ in
    #                                             range(len(target_names))]

    gating_networks = [SimpleLSTM(torch.zeros(size=(1, 768)), args.hidden_dim_g, args.n_experts,
                                                           device=args.device, use_lengths=args.use_lengths,
                                  use_bert_embeds=True) for _ in
                                                range(len(target_names))]

    shared_layers = [MultitaskConvNet(input_channels=1, filter_list=args.filter_list_experts,
                    embed_matrix=torch.zeros(size=(1, 768)), num_filters=args.num_filters_experts,
                                      use_bert_embeds=True)
                       for _ in range(args.n_experts)]

    model = MultiGateMixtureofExperts(shared_layers=shared_layers, gating_networks=gating_networks,
                                      towers=towers, device=args.device, include_lens=args.use_lengths,
                                      batch_size=args.batch_size, gating_drop=args.gate_dropout,
                                      mean_diff=args.mean_diff, weight_adjust_mode=args.balancing_strategy)

    # TODO: c
    if args.class_weighting:
        task_weights = multitask_class_weighting(train_set, target_names, output_dimensions)
        losses = {name: nn.CrossEntropyLoss(weight=task_weights[name].to(args.device)) for name in target_names}
    else:
        losses = {name: nn.CrossEntropyLoss() for name in target_names}

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma)

    train(model, losses, optimizer, scheduler, train_set, device=args.device,
          include_lengths=args.use_lengths, save_path=args.logdir, save_name="%s_datasets" % "_".join(target_names),
          tensorboard_dir=args.logdir+"/runs", n_epochs=args.n_epochs, checkpoint_interval=args.save_interval,
          clip_val=args.gradient_clip, balancing_epoch_num=args.balance_epoch_cnt)

    print("Evaluating model")
    model.load_state_dict(torch.load("%s/%s_datasets_epoch_%d.pt" % (args.logdir, "_".join(target_names),
                                                                                              args.n_epochs-1)))
    evaluation(model, test_set, losses, device=args.device)


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
    parser.add_argument("--adaptive_gate_dropout", type=int, default=0)
    parser.add_argument("--balancing_strategy", type=str, default=None)
    parser.add_argument("--mean_diff", type=float, default=0.1)
    parser.add_argument("--balance_epoch_cnt", type=int, default=0)
    args = parser.parse_args()

    args.use_lengths = eval(args.use_lengths)
    args.do_lowercase = eval(args.do_lowercase)
    args.linear_layers = list(map(int, args.linear_layers))
    args.filter_list_g = list(map(int, args.filter_list_g))
    args.filter_list_experts = list(map(int, args.filter_list_experts))

    main(args)



