import torch
import shutil
from tqdm import tqdm
from codebase.experiments.single_task_classification.config import *
from codebase.data_classes.dataiterator import DataIterator
from torchtext.data import BucketIterator
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from scipy.stats import variation

def train(model, criterion, optimizer, scheduler, dataset, n_epochs=5, device=torch.device("cpu"), include_lengths=False,
           save_path=None, save_name=None, use_tensorboard=False, checkpoint_interval=5):
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
        epoch_acc = 0

        for i, batch in tqdm(enumerate(dataset)):
            batch_running_loss = 0.0
            optimizer.zero_grad()
            X, y, _ = batch
            # Whether the padding should be removed when fed into the LSTM
            outputs = model(X)

            loss = criterion(outputs, y)
            batch_running_loss += loss.item()

            # Calculating several batch statistics
            batch_correct = (torch.max(outputs, 1)[1].view(y.size()).data == y.data).sum().item()
            batch_f1 = f1_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
            batch_recall = recall_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
            batch_precision = precision_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
            batch_acc = accuracy_score(y.cpu(), outputs.detach().cpu().argmax(1))
            epoch_acc += batch_acc
            epoch_recall += batch_recall
            epoch_precision += batch_precision
            epoch_f1 += batch_f1
            epoch_running_loss += batch_running_loss
            epoch_correct += batch_correct
            epoch_total += outputs.shape[0]
            # training the network
            loss.backward()
            optimizer.step()
        prog_string = "[|Train| Loss: %.3f, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                      % (epoch_running_loss, epoch_correct / epoch_total,
                         epoch_f1 / (i+1), epoch_recall / (i+1), epoch_precision / (i+1))
        with open("%s/results.txt" % save_path, "a") as f:
            f.write(prog_string+"\n")

    print('Finished Training')
    torch.save(model.state_dict(), "%s/%s_epoch_%d.pt" % (save_path, save_name, epoch))
    #print(model.softmax(model.gating_network(inputs, lengths=lengths)).unsqueeze(1))
    return model












def evaluation(model, dataset, criterion, include_lengths=True, device=None):
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
        X, y, _ = batch
        outputs = model(X)

        loss = criterion(outputs, y)
        epoch_running_loss += loss.item()
        # Calculate several batch statistics
        batch_correct = (torch.max(outputs, 1)[1].view(y.size()).data == y.data).sum().item()
        batch_f1 = f1_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
        batch_recall = recall_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
        batch_precision = precision_score(y.cpu(), outputs.detach().cpu().argmax(1), average='micro')
        epoch_recall += batch_recall
        epoch_precision+=batch_precision
        epoch_correct += batch_correct
        epoch_f1 += batch_f1
        epoch_total += outputs.shape[0]

    prog_string = "[|Test| Loss: %.3f, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                  %(epoch_running_loss, epoch_correct/epoch_total,
                                                           epoch_f1/(i+1), epoch_recall/(i+1), epoch_precision/(i+1))
    print(prog_string)














def train_moe(model, criterion, optimizer, scheduler, dataset, n_epochs=5, device=torch.device("cpu"), include_lengths=False,
           save_path=None, save_name=None, use_tensorboard=False):
    # Set the model in training mode just to be safe

    model.to(device)
    model.train()
    for epoch in range(n_epochs):
        if save_path:
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
            X, y, _ = batch
            # Whether the padding should be removed when fed into the LSTM
            if include_lengths:
                inputs, lengths = X
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                #TODO REMOVE IF WEIGHTING IS NOT USED
                outputs, weights = model(inputs, lengths)
            else:
                X = X.to(device)

                outputs, weights = model(X)

            weights_loss = torch.from_numpy(variation(torch.sum(weights.detach().cpu(), dim=0), axis=1)).cuda()
            loss = criterion(outputs, y) + 30*(weights_loss**2)
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
                         epoch_f1 / i, epoch_recall / i, epoch_precision / i)
        with open("%s/results.txt" % save_path, "a") as f:
            f.write(prog_string+"\n")

    print('Finished Training')
    #print(model.softmax(model.gating_network(inputs, lengths=lengths)).unsqueeze(1))
    return model










def evaluation_moe(model, dataset, criterion, include_lengths=True, device=None):
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
        X, y, _ = batch
        if include_lengths:
            inputs, lengths = X
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            outputs, weights = model(inputs, lengths)
        else:
            X = X.to(device)
            outputs, weights = model(X)
        print(weights)
        quit()
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
                                                           epoch_f1/i, epoch_recall/i, epoch_precision/i)
    print(prog_string)
