import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from codebase.data.synthdata import SynthData
from codebase.models.simplemoe import SimpleMoE
from codebase.models.mlp import MLP
from codebase.models.convnet import ConvNet
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# This file contains a generic training function for training pytorch networks


def train(model, criterion, optimizer, dataset, n_epochs=10, device=torch.device("cpu")):
    # Set the model in training mode ust to be safe
    model = model.to(device)
    model = model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataset):
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            # reset gradients
            optimizer.zero_grad()
            # calculate the loss
            outputs = model(X)
            loss = criterion(outputs, y.squeeze())

            # training the network
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    return model


def eval(model, test_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))




# create the dataset
# num_datapoints = 1024
# num_classes = 5
# num_features = 12
# num_experts = 3
#
# synthetic_data = SynthData(num_points=num_datapoints, num_classes=num_classes,
#                            num_features=num_features)
#
# dataloader = DataLoader(synthetic_data, batch_size=4,
#                         shuffle=True, num_workers=0) # weird error when setting num_workers > 0
#
#
# g = MLP(num_features, [16, 32, 16], num_experts)
# experts = [MLP(num_features, [64, 128, 64], num_classes) for x in range(num_experts)]
#
# model = SimpleMoE(input_dim=num_features, gating_network=g, expert_networks=experts,
#                   output_dim=num_classes)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_datapoints = 1024
num_classes = 10
num_features = 12
num_experts = 5

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)

dataiter = iter(trainloader)
images, labels = dataiter.next()

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=0)

g = ConvNet(3, num_experts)
# When doing it like this, they have to be in a ModuleList, otherwise the optimizer will
# not be able to see them!
experts = [ConvNet(3, num_classes) for x in range(num_experts)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = SimpleMoE(input_dim=num_features, gating_network=g, expert_networks=experts,
                  output_dim=num_classes, device=device)

writer = SummaryWriter('runs/convnet_setup')



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model = train(model, criterion, optimizer, trainloader, device=device)
eval(model, test_loader=testloader)
writer.add_graph(model, images.to(device))
writer.close()


