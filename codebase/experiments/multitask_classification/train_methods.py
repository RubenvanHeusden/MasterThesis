import torch
import shutil
import io
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# TODO: towers should be a dict of shape "{"example_task_tower": tower}"
def train(model, criterion_dict, optimizer, scheduler, dataset, n_epochs=5, device=torch.device("cpu"),
          save_path=None, save_name=None, tensorboard_dir=False, checkpoint_interval=5,
          include_lengths=True, clip_val=0, balancing_epoch_num=0):
    # Set the model in training mode just to be safe
    torch.cuda.empty_cache()
    model.to(device)
    model.train()

    if tensorboard_dir:
        writer = SummaryWriter(tensorboard_dir)
        # if include_lengths:
        #     s = dataset.sample().cuda()
        #     sample = s, torch.tensor([s.shape[0]]).cuda()
        # else:
        #     sample = dataset.sample().cuda().unsqueeze(0)
        # writer.add_graph(model, [sample])

    for epoch in range(n_epochs):

        if balancing_epoch_num and (epoch == balancing_epoch_num):
            model.weight_adjust_mode = None

        if save_path:
            if epoch % checkpoint_interval == 0:
                torch.save(model.state_dict(), "%s/%s_epoch_%d.pt" % (save_path, save_name, epoch))
        epoch_weights = defaultdict(list)
        all_predictions = defaultdict(list)
        all_ground_truth_labels = defaultdict(list)
        epoch_running_loss = defaultdict(float)
        # Calculate several training statistics
        for i, batch in tqdm(enumerate(dataset)):
            optimizer.zero_grad()
            X, *targets, tasks = batch

            for y in range(len(targets)):
                targets[y] = targets[y].to(device)

            if isinstance(X, tuple):
                X = list(X)
                for z in range(len(X)):
                    X[z] = X[z].to(device)
            else:
                X = X.to(device)

            loss = 0

            if model.return_weights:
                outputs, weights = model(X, tasks)
                for w in range(len(tasks)):
                    epoch_weights[tasks[w]].append(weights[w].detach().cpu())
            else:
                outputs = model(X, tasks)
            for p in range(len(tasks)):
                all_predictions[tasks[p]].extend(outputs[p].detach().cpu().argmax(1).tolist())
                # see if we need to even this out
                loss += criterion_dict[tasks[p]](outputs[p], targets[p])

            # training the network
            loss.backward()
            if clip_val:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

            optimizer.step()

            for t in range(len(tasks)):
                all_ground_truth_labels[tasks[t]].extend(targets[t].cpu().tolist())
                epoch_running_loss[tasks[t]] += loss.item()
        scheduler.step()
        for task in all_predictions.keys():
            correct_list = [1 if a == b else 0 for a, b in zip(all_predictions[task], all_ground_truth_labels[task])]
            print("Task %s" % task)
            print(sum([1 for item in all_predictions[task] if item != 0]))
            acc = sum(correct_list) / len(correct_list)
            prog_string = "[|Train| Task: %s Loss: %.3f, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                          % (task, epoch_running_loss[task], acc,
                     f1_score(all_ground_truth_labels[task], all_predictions[task], average="weighted"),
                     recall_score(all_ground_truth_labels[task], all_predictions[task], average="weighted"),
                     precision_score(all_ground_truth_labels[task], all_predictions[task], average="weighted"))
            with open("%s/results.txt" % save_path, "a") as f:
                f.write(prog_string+"\n")
            if tensorboard_dir:
                writer.add_scalar('loss_%s' % task, epoch_running_loss[task], epoch)
                writer.add_scalar('accuracy_%s' % task, acc, epoch)
                if model.return_weights:
                    epoch_weights[task] = torch.cat(epoch_weights[task], dim=0)

                    task_mean = epoch_weights[task].mean(0).tolist()[0]

                    expert_weights = epoch_weights[task].squeeze().numpy()
                    task_classes_df = pd.DataFrame(expert_weights,
                                 columns=range(expert_weights.shape[1]))
                    task_classes_df['class'] = all_ground_truth_labels[task]

                    figure = plt.figure()
                    ax = figure.add_subplot(111)
                    task_classes_df.groupby('class').mean().plot(kind='bar', ax=ax)
                    writer.add_figure("weight distribution classes for Task %s" % task,
                                      figure, epoch)

                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(111)

                    y_pos = np.arange(len(task_mean))
                    ax2.bar(y_pos, task_mean)
                    writer.add_figure("Average distribution of experts %s" % task,
                                      fig2, epoch)
    print('Finished Training')
    torch.save(model.state_dict(), "%s/%s_epoch_%d.pt" % (save_path, save_name, epoch))
    #print(model.softmax(model.gating_network(inputs, lengths=lengths)).unsqueeze(1))
    if tensorboard_dir:
        writer.close()
    return model


def evaluation(model, dataset, criterion=None, device=None):
    model.to(device)
    # Set the model to evaluation mode, important because of the Dropout Layers
    model = model.eval()
    # Calculate several test statistics
    epoch_weights = defaultdict(list)
    all_predictions = defaultdict(list)
    all_ground_truth_labels = defaultdict(list)
    # Calculate several training statistics
    for i, batch in tqdm(enumerate(dataset)):
        X, *targets, tasks = batch

        for y in range(len(targets)):
            targets[y] = targets[y].to(device)

        if isinstance(X, tuple):
            X = list(X)
            for z in range(len(X)):
                X[z] = X[z].to(device)
        else:
            X = X.to(device)

        if model.return_weights:
            outputs, weights = model(X, tasks)
            for w in range(len(tasks)):
                epoch_weights[tasks[w]].append(weights[w].detach().cpu())
        else:
            outputs = model(X, tasks)
        for p in range(len(tasks)):
            all_predictions[tasks[p]].extend(outputs[p].detach().cpu().argmax(1).tolist())
            all_ground_truth_labels[tasks[p]].extend(targets[p].cpu().tolist())
            # see if we need to even this out

    for task in all_predictions.keys():
        correct_list = [1 if a == b else 0 for a, b in zip(all_predictions[task], all_ground_truth_labels[task])]
        acc = sum(correct_list) / len(correct_list)
        prog_string = "[|Train| Task: %s, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                      % (task, acc,
                 f1_score(all_ground_truth_labels[task], all_predictions[task], average="weighted"),
                 recall_score(all_ground_truth_labels[task], all_predictions[task], average="weighted"),
                 precision_score(all_ground_truth_labels[task], all_predictions[task], average="weighted"))
        print(prog_string)