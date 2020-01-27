import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, LabelField
from torchtext.datasets import IMDB
from torchtext.data import BucketIterator, Iterator
from codebase.models.simplelstm import SimpleLSTM
from torch.optim.lr_scheduler import StepLR

# First we will load in a dataset, for now we will use IMDB
# as it is included in torchtext
# TODO: clip gradients


class DataIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            yield batch.text, batch.label

    def __len__(self):
        return len(self.dataloader)


# Define the labels for the train and test text
# We use include_lengths=True so that we can remove the padding for the LSTM module

batch_size = 128
include_lens = True
TEXT = Field(lower=True, include_lengths=include_lens, batch_first=True)
# TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=True, batch_first=True)
LABEL = LabelField(dtype=torch.long)

# specify the device that we want to use
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Load the IMDB dataset and split it into train and test portions
print("----- Loading the dataset ------")
# Could set max_size here for faster initialization
train, test = IMDB.splits(TEXT, LABEL)
print("----- Text has been loaded in and preprocessed -----")
# Load the glove vectors and initialize the vocabulary with the vectors
print("----- Loading the word embeddings -----")
TEXT.build_vocab(train, vectors="glove.6B.300d")
LABEL.build_vocab(train)
print("----- Word Embeddings have been loaded -----")


train_iter = BucketIterator(
    train,
    batch_size=batch_size,
    device=device,
    sort_within_batch=False,
    sort_key=lambda x: len(x.comment_text),
    repeat=False
)

test_iter = Iterator(test, batch_size=batch_size,
                     device=device, sort=False,
                     sort_within_batch=False,
                     repeat=False)

# Make a separate Bool for the include_lengths option of the dataset


lstm_model = SimpleLSTM(vocab=TEXT.vocab, embedding_dim=300, hidden_dim=128, output_dim=2, device=device)
# TODO: check dimensions
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lstm_model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)


print("----- Starting Training -----")


def train(model, criterion, optimizer, dataset, n_epochs=5, device=torch.device("cpu"), include_lengths=False):
    # Set the model in training mode ust to be safe
    model = model.to(device)
    model = model.train()
    for epoch in range(n_epochs):
        for i, batch in enumerate(dataset):
            optimizer.zero_grad()
            X, y = batch
            if include_lengths:
                inputs, lengths = X
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                outputs = model(inputs, lengths)
            else:
                X = X.to(device)
                # reset gradients
                # calculate the loss
                outputs = model(X)

            loss = criterion(outputs, y)
            print("Epoch: %d Batch %d of %d " % (epoch+1, i+1, len(dataset)))
            num_corrects = (torch.max(outputs, 1)[1].view(y.size()).data == y.data).sum()
            acc = torch.div(num_corrects.cpu(), float(batch_size))
            # training the network
            loss.backward()
            optimizer.step()
            # print statistics
            # print every 2000 mini-batches
        scheduler.step()
    print('Finished Training')
    return model


train(lstm_model, criterion, optimizer, DataIterator(train_iter), device=device, include_lengths=include_lens)
