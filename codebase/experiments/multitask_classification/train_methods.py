import torch
import shutil
from tqdm import tqdm
from codebase.experiments.single_task_classification.config import *
from codebase.data_classes.dataiterator import DataIterator
from torchtext.data import BucketIterator
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from itertools import cycle, repeat


# TODO: towers should be a dict of shape "{"example_task_tower": tower}"
def train(model, criterion, optimizer, scheduler, dataset, n_epochs=5, device=torch.device("cpu"),
          save_path=None, save_name=None, use_tensorboard=False, checkpoint_interval=5,
          include_lengths=True):
    # Set the model in training mode just to be safe

    model.to(device)
    model.train()
    for epoch in range(n_epochs):
        if save_path:
            if epoch % checkpoint_interval == 0:
                torch.save(model.state_dict(), "%s/%s_epoch_%d.pt" % (save_path, save_name, epoch))
            with open("%s/model_params.txt" % save_path, "w") as f_obj:
                f_obj.write(str(model.params))
            shutil.copyfile("config.py", "%s/config.py" % save_path)

        # Calculate several training statistics
        epoch_running_loss = 0.0
        epoch_precision = 0
        epoch_recall = 0
        epoch_correct = 0
        epoch_total = 0
        epoch_f1 = 0
        for i, batch in tqdm(enumerate(dataset)):
            batch_running_loss = 0.0
            optimizer.zero_grad()
            X, y, task = batch
            outputs = model(X, task)

            loss = criterion(outputs, y)
            batch_running_loss += loss.item()

            # Calculating several batch statistics
            batch_correct = (torch.max(outputs, 1)[1].view(y.size()).data == y.data).sum().item()
            batch_f1 = f1_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
            batch_recall = recall_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
            batch_precision = precision_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
            epoch_recall += batch_recall
            epoch_precision += batch_precision
            epoch_f1+=batch_f1
            epoch_running_loss += batch_running_loss
            epoch_correct += batch_correct
            epoch_total += outputs.shape[0]
            # training the network
            loss.backward()
            optimizer.step()
            # print("[Epoch: %d, Batch: %d, Loss: %.3f, Acc: %.3f]" % (epoch + 1, i+1, batch_running_loss,
            #                                                          batch_correct / batch_total))
            # print statistics
            # print every 2000 mini-batches
        scheduler.step()
        prog_string = "[|Train| Loss: %.3f, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                      % (epoch_running_loss, epoch_correct / epoch_total,
                         epoch_f1 / (i+1), epoch_recall / (i+1), epoch_precision / (i+1))
        with open("%s/results.txt" % save_path, "a") as f:
            f.write(prog_string+"\n")

    print('Finished Training')
    torch.save(model.state_dict(), "%s/%s_epoch_%d.pt" % (save_path, save_name, epoch))
    #print(model.softmax(model.gating_network(inputs, lengths=lengths)).unsqueeze(1))
    return model


def evaluation(model, dataset, criterion, device=None):
    model.to(device)
    # Set the model to evaluation mode, important because of the Dropout Layers
    model = model.eval()
    # Calculate several test statistics
    epoch_running_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    epoch_f1 = 0
    epoch_precision = 0
    epoch_recall = 0
    for i, batch in tqdm(enumerate(dataset)):
        X, y, task = batch
        if isinstance(X, tuple):
            X = list(X)
            for k in range(len(X)):
                X[k] = X[k].to(device)
        else:
            X = X.to(device)
        outputs = model(X, task)

        loss = criterion(outputs, y)
        epoch_running_loss += loss.item()
        # Calculate several batch statistics
        batch_correct = (torch.max(outputs, 1)[1].view(y.size()).data == y.data).sum().item()
        batch_f1 = f1_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
        batch_recall = recall_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
        batch_precision = precision_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
        epoch_recall += batch_recall
        epoch_precision += batch_precision
        epoch_correct += batch_correct
        epoch_f1 += batch_f1
        epoch_total += outputs.shape[0]

    prog_string = "[|Test| Loss: %.3f, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                  %(epoch_running_loss, epoch_correct/epoch_total,
                                                           epoch_f1/(i+1), epoch_recall/(i+1), epoch_precision/(i+1))
    print(prog_string)

