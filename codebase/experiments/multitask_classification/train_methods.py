import torch
import shutil
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


# TODO: towers should be a dict of shape "{"example_task_tower": tower}"
def train(model, criterion, optimizer, scheduler, dataset, n_epochs=5, device=torch.device("cpu"),
          save_path=None, save_name=None, tensorboard_dir=False, checkpoint_interval=5,
          include_lengths=True, clip_val=0):
    # Set the model in training mode just to be safe

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
        if save_path:
            if epoch % checkpoint_interval == 0:
                torch.save(model.state_dict(), "%s/%s_epoch_%d.pt" % (save_path, save_name, epoch))
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
            for p, task in enumerate(tasks):
                outputs = model(X, task)
                all_predictions[task].extend(outputs.detach().cpu().argmax(1).tolist())
                # see if we need to even this out
                loss += criterion(outputs, targets[p])

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

        for i, task in enumerate(tasks):
            outputs = model(X, task)
            # see if we need to even this out
            # training the network
            all_predictions[task].extend(outputs.detach().cpu().argmax(1).tolist())

        for t in range(len(tasks)):
            all_ground_truth_labels[tasks[t]].extend(targets[t].cpu().tolist())

    for task in all_predictions.keys():
        correct_list = [1 if a == b else 0 for a, b in zip(all_predictions[task], all_ground_truth_labels[task])]
        acc = sum(correct_list) / len(correct_list)
        prog_string = "[|Train| Task: %s, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                      % (task, acc,
                 f1_score(all_ground_truth_labels[task], all_predictions[task], average="weighted"),
                 recall_score(all_ground_truth_labels[task], all_predictions[task], average="weighted"),
                 precision_score(all_ground_truth_labels[task], all_predictions[task], average="weighted"))
        print(prog_string)