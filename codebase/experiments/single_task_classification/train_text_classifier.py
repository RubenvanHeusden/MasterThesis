import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data import Field, LabelField
from codebase.data_classes.imdbdataset import IMDBDataset
from codebase.data_classes.sttdataset import SSTDataset
from codebase.data_classes.customdataloader import CustomDataLoader
from codebase.data_classes.yelpdataset import YelpDataset
from codebase.models.simplelstm import SimpleLSTM
from torch.optim.lr_scheduler import StepLR
from codebase.experiments.single_task_classification.train_methods import *
from codebase.experiments.single_task_classification.config import *
from codebase.models.simplemoe import SimpleMoE
from codebase.models.mlp import MLP


# TODO: clip gradients
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
#device = torch.device("cpu")
TEXT = Field(lower=SET_LOWERCASE, include_lengths=include_lens, batch_first=True)
# TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=True, batch_first=True)
LABEL = LabelField(dtype=torch.long)
print("--- Starting with the SST dataset ---")
#dataset = SSTDataset(TEXT, LABEL, path="../.data/imdb/aclImdb").load()
dataset = SSTDataset(TEXT, LABEL).load()
#dataset = YelpDataset(TEXT, LABEL, path="../.data/yelp").load()

print("--- Finished with the SST dataset ---")
# Load the IMDB dataset and split it into train and test portions
dloader = CustomDataLoader(dataset, TEXT, LABEL)
data_iterators = dloader.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                             batch_size=BATCH_SIZE, device=device)


# Some sample models that can be used are listed below, uncomment the particular model to use it
lstm_model = SimpleLSTM(vocab=TEXT.vocab.vectors, embedding_dim=300, hidden_dim=64, output_dim=2, device=device,
                        use_lengths=include_lens)
# g = SimpleLSTM(vocab=TEXT.vocab.vectors, embedding_dim=300, hidden_dim=4, output_dim=2, device=device)
# expert_networks = [SimpleLSTM(vocab=TEXT.vocab.vectors, embedding_dim=300, hidden_dim=64, output_dim=5, device=device)
#                    for _ in range(2)]
#
# moe_model = SimpleMoE(gating_network=g, expert_networks=
# expert_networks, output_dim=2, device=device)
#

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lstm_model.parameters(), lr=OPTIMIZER_LEARNING_RATE)
#optimizer = optim.SGD(moe_model.parameters(), lr=OPTIMIZER_LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

# see if this fixes memory errors


train(lstm_model, criterion, optimizer, scheduler, data_iterators[0], device=device, include_lengths=include_lens,
    save_path='saved_models/MoE', save_name="SST_dataset", use_tensorboard=True, n_epochs=50)

print("Evaluating model")
lstm_model.load_state_dict(torch.load("saved_models/MoE/SST_dataset_epoch_49.pt"))
evaluation(lstm_model, data_iterators[2], criterion, device=device, include_lengths=include_lens)