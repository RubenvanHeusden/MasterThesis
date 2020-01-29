import torch
import shutil
from tqdm import tqdm
from codebase.experiments.single_task_classification.config import *


def train(model, criterion, optimizer, scheduler, dataset, n_epochs=5, device=torch.device("cpu"), include_lengths=False,
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
        epoch_running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for i, batch in tqdm(enumerate(dataset)):
            batch_running_loss = 0.0
            batch_correct = 0
            batch_total = float(BATCH_SIZE)
            optimizer.zero_grad()
            X, y = batch
            if include_lengths:
                inputs, lengths = X
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                outputs = model(inputs, lengths)
            else:
                X = X.to(device)

                outputs = model(X)

            loss = criterion(outputs, y)
            batch_running_loss += loss.item()
            batch_correct = (torch.max(outputs, 1)[1].view(y.size()).data == y.data).sum().item()
            epoch_running_loss += batch_running_loss
            epoch_correct += batch_correct
            epoch_total += batch_total
            # training the network
            loss.backward()
            optimizer.step()
            # print("[Epoch: %d, Batch: %d, Loss: %.3f, Acc: %.3f]" % (epoch + 1, i+1, batch_running_loss,
            #                                                          batch_correct / batch_total))
            # print statistics
            # print every 2000 mini-batches
        scheduler.step()
        prog_string = "[Epoch: %d, Loss: %.3f, Acc: %.3f]" % (epoch+1, epoch_running_loss, epoch_correct/epoch_total)
        with open("%s/results.txt" % save_path, "a") as f:
            f.write(prog_string+"\n")
    print('Finished Training')
    return model


def evaluation(model, dataset, criterion, include_lengths=True, device=None):
    model.to(device)
    model.eval()
    epoch_running_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    for i, batch in tqdm(enumerate(dataset)):
        X, y = batch
        if include_lengths:
            inputs, lengths = X
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            outputs = model(inputs, lengths)
        else:
            X = X.to(device)
            outputs = model(X)

        loss = criterion(outputs, y)
        epoch_running_loss += loss.item()
        batch_correct = (torch.max(outputs, 1)[1].view(y.size()).data == y.data).sum().item()
        epoch_correct += batch_correct
        epoch_total += BATCH_SIZE

    prog_string = "[|Test| Loss: %.3f, Acc: %.3f]" % (epoch_running_loss, epoch_correct/epoch_total)
    print(prog_string)

