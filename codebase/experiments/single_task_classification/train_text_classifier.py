import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data import Field, LabelField
from codebase.data.imdbdataset import IMDBDataset
from codebase.data.customdataloader import CustomDataLoader
from codebase.models.simplelstm import SimpleLSTM
from torch.optim.lr_scheduler import StepLR
from codebase.experiments.single_task_classification.train_methods import *
from codebase.experiments.single_task_classification.config import *
from codebase.models.simplemoe import SimpleMoE
from codebase.models.mlp import MLP

# TODO: clip gradients
# TODO: make the lengths_feature an attribute to the LSTM

# Set the random seed for experiments (check if I need to do this for all the other files as well)
torch.cuda.empty_cache()
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# TODO: when doing a 'real' experiment, uncomment the two lines below!
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
batch_size = BATCH_SIZE
include_lens = INCLUDE_LENGTHS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
TEXT = Field(lower=SET_LOWERCASE, include_lengths=include_lens, batch_first=True)
# TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=True, batch_first=True)
LABEL = LabelField(dtype=torch.long)

dataset = IMDBDataset(TEXT, LABEL, path="../.data/imdb/aclImdb").load()
# Load the IMDB dataset and split it into train and test portions
dloader = CustomDataLoader(dataset, TEXT, LABEL)
data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                             batch_size=BATCH_SIZE, device=device)


# Some sample models that can be used are listed below, uncomment the particular model to use it
# lstm_model = SimpleLSTM(vocab=TEXT.vocab, embedding_dim=300, hidden_dim=128, output_dim=2, device=device)
g = SimpleLSTM(vocab=TEXT.vocab, embedding_dim=300, hidden_dim=8, output_dim=3, device=device)
expert_networks = [SimpleLSTM(vocab=TEXT.vocab, embedding_dim=300, hidden_dim=128, output_dim=2, device=device)
                   for _ in range(3)]

moe_model = SimpleMoE(None, gating_network=g, expert_networks=
                      expert_networks, output_dim=2, device=torch.device("cpu"))


criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(lstm_model.parameters(), lr=OPTIMIZER_LEARNING_RATE)
optimizer = optim.SGD(moe_model.parameters(), lr=OPTIMIZER_LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

# see if this fixes memory errors


train(moe_model, criterion, optimizer, scheduler, data_iterators[0], device=device, include_lengths=include_lens,
     save_path='saved_models/MoE', save_name="IMDB_dataset", use_tensorboard=True)

# print("Evaluating model")
# lstm_model.load_state_dict(torch.load("saved_models/LSTM/IMDB_dataset_epoch_4.pt"))
# evaluation(lstm_model, data_iterators[1], criterion, device=device)