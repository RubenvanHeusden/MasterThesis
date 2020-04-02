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
          save_path=None, save_name=None, use_tensorboard=False, checkpoint_interval=5,
          include_lengths=True, clip_val=0):
    # Set the model in training mode just to be safe

    model.to(device)
    model.train()

    # if tensorboard_dir:
    #     writer = SummaryWriter(tensorboard_dir)
    #     if include_lengths:
    #         s = dataset.sample().cuda()
    #         sample = s, torch.tensor([s.shape[0]]).cuda()
    #     else:
    #         sample = dataset.sample().cuda().unsqueeze(0)
    #     writer.add_graph(model, [sample])

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
            for y in targets:
                y = y.to(device)
            if isinstance(X, tuple):
                X = list(X)
                for z in range(len(X)):
                    X[z] = X[z].to(device)
            else:
                X = X.to(device)
            loss = 0
            for i, task in enumerate(tasks):
                outputs = model(X, task)
                # see if we need to even this out
                loss += criterion(outputs, targets[i])
            # training the network
            loss.backward()
            if clip_val:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
            quit()
            all_predictions[task].extend(outputs.detach().cpu().argmax(1).tolist())
            all_ground_truth_labels[task].extend(y.cpu().tolist())
            epoch_running_loss[task] += loss.item()
            # Calculating several batch statistics

        scheduler.step()
        for task in all_predictions.keys():
            correct_list = [1 if a == b else 0 for a, b in zip(all_predictions[task], all_ground_truth_labels[task])]
            acc = sum(correct_list) / len(correct_list)
            prog_string = "[|Train| Task: %s Loss: %.3f, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                          % (task, epoch_running_loss[task], acc,
                     f1_score(all_ground_truth_labels[task], all_predictions[task], average="micro"),
                     recall_score(all_ground_truth_labels[task], all_predictions[task], average="micro"),
                     precision_score(all_ground_truth_labels[task], all_predictions[task], average="micro"))
            with open("%s/results.txt" % save_path, "a") as f:
                f.write(prog_string+"\n")
        with open("%s/results.txt" % save_path, "a") as f:
            f.write("\n")
    print('Finished Training')
    torch.save(model.state_dict(), "%s/%s_epoch_%d.pt" % (save_path, save_name, epoch))
    #print(model.softmax(model.gating_network(inputs, lengths=lengths)).unsqueeze(1))
    return model


def evaluation(model, dataset, criterion, device=None):
    model.to(device)
    # Set the model to evaluation mode, important because of the Dropout Layers
    model = model.eval()
    # Calculate several test statistics
    epoch_running_loss = defaultdict(float)
    all_predictions = defaultdict(list)
    all_ground_truth_labels = defaultdict(list)
    for i, batch in enumerate(dataset):
        X, y, task = batch
        y = y.to(device)
        if isinstance(X, tuple):
            X = list(X)
            for z in range(len(X)):
                X[z] = X[z].to(device)
        else:
            X = X.to(device)
        outputs = model(X, task)

        loss = criterion(outputs, y)
        epoch_running_loss[task] += loss.item()
        all_predictions[task].extend(outputs.detach().cpu().argmax(1).tolist())
        all_ground_truth_labels[task].extend(y.cpu().tolist())
       # Calculate several batch statistics
    for task in all_predictions.keys():
        correct_list = [1 if a == b else 0 for a, b in zip(all_predictions[task], all_ground_truth_labels[task])]
        acc = sum(correct_list) / len(correct_list)
        prog_string = "[|Train| Task: %s Loss: %.3f, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                      % (task, epoch_running_loss[task], acc,
                 f1_score(all_ground_truth_labels[task], all_predictions[task], average="micro"),
                 recall_score(all_ground_truth_labels[task], all_predictions[task], average="micro"),
                 precision_score(all_ground_truth_labels[task], all_predictions[task], average="micro"))
        print(prog_string)

