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
from codebase.experiments.multitask_classification.train_methods import *
from codebase.experiments.multitask_classification.config import *
from codebase.models.simplemoe import SimpleMoE
from codebase.models.multitaskmodel import MultiTaskModel
from codebase.models.multitasklstm import MultiTaskLSTM
from codebase.models.mlp import MLP
import itertools

# TODO: clip gradients
# for the multitask learning, make a dictionary containing "task": data

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
TEXT_SST = Field(lower=SET_LOWERCASE, include_lengths=include_lens, batch_first=True)
# TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="en", include_lengths=True, batch_first=True)
LABEL_SST = LabelField(dtype=torch.long)
TEXT_YELP = Field(lower=SET_LOWERCASE, include_lengths=include_lens, batch_first=True)
LABEL_YELP = LabelField(dtype=torch.long)
print("--- Starting with the SST dataset ---")
#dataset = SSTDataset(TEXT, LABEL, path="../.data/imdb/aclImdb").load()
dataset_sst = SSTDataset(TEXT_SST, LABEL_SST).load()
dataset_yelp = YelpDataset(TEXT_YELP, LABEL_YELP, path="../.data/yelp").load()
print("--- Finished with the SST dataset ---")
# Load the IMDB dataset and split it into train and test portions
dloader_sst = CustomDataLoader(dataset_sst, TEXT_SST, LABEL_SST, task_name="sst")
dloader_yelp = CustomDataLoader(dataset_yelp, TEXT_YELP, LABEL_YELP, task_name="yelp")
data_iterators_sst = dloader_sst.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                             batch_size=BATCH_SIZE, device=device)
data_iterators_yelp = dloader_yelp.construct_iterators(vectors="glove.6B.300d", vector_cache="../.vector_cache",
                                             batch_size=BATCH_SIZE, device=device)

#print(*data_iterators_yelp)
total_vocab = torch.cat((TEXT_SST.vocab.vectors, TEXT_YELP.vocab.vectors))
train_iterators = itertools.chain(*zip(*(data_iterators_sst[0], data_iterators_yelp[0])))
test_iterators = itertools.chain(*zip(*(data_iterators_sst[1], data_iterators_yelp[1])))
# Some sample models that can be used are listed below, uncomment the particular model to use it

hidden_repr = 64
yelp_tower = MLP(hidden_repr, [24], 5, name="yelp")
sst_tower = MLP(hidden_repr, [24], 5, name="sst")
towers = [sst_tower, yelp_tower]
lstm_model = MultiTaskLSTM(vocab=total_vocab, embedding_dim=300, hidden_dim=hidden_repr, device=device,
                        use_lengths=include_lens)

multitask_model = MultiTaskModel(shared_layer=lstm_model, tower_list=towers, batch_size=BATCH_SIZE,
                                 input_dimension=300, device=device, include_lens=INCLUDE_LENGTHS)

# g = SimpleLSTM(vocab=TEXT.vocab, embedding_dim=300, hidden_dim=4, output_dim=2, device=device)
# expert_networks = [SimpleLSTM(vocab=TEXT.vocab, embedding_dim=300, hidden_dim=64, output_dim=5, device=device)
#                    for _ in range(2)]

# moe_model = SimpleMoE(None, gating_network=g, expert_networks=
# expert_networks, output_dim=2, device=device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(multitask_model.parameters(), lr=OPTIMIZER_LEARNING_RATE)
#optimizer = optim.SGD(moe_model.parameters(), lr=OPTIMIZER_LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

# see if this fixes memory errors


train(multitask_model, criterion, optimizer, scheduler, list(train_iterators), device=device, include_lengths=include_lens,
    save_path='saved_models/LSTM', save_name="SST_dataset", use_tensorboard=True, n_epochs=50)

print("Evaluating model")
multitask_model.load_state_dict(torch.load("saved_models/LSTM/SST_dataset_epoch_49.pt"))
evaluation(multitask_model, data_iterators_sst[1], criterion, device=device, include_lengths=include_lens)
evaluation(multitask_model, data_iterators_yelp[1], criterion, device=device, include_lengths=include_lens)