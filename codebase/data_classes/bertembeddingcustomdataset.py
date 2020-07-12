import torch
from torch.utils.data import Dataset, DataLoader
import typing


class BertEmbeddingCustomDataset(Dataset):
    def __init__(self, f_path: str = "../.data/custom_data_bert/", mode="train"):
        self.target_names = ("label", "intent_classes", "emotion_classes")
        if mode == "train":
            self.embeddings, self.lengths, self.cat, self.intent, self.emotion = torch.load(f_path+"train.pt")
        elif mode == "test":
            self.embeddings, self.lengths, self.cat, self.intent, self.emotion = torch.load(f_path+"test.pt")
        else:
            raise(Exception("It seems that the selected dataloading mode is invalid"))

        self.emotion_dict = {'gratitude': 4, 'anger_agitation': 1, 'hope_anticipation': 2, 'concern': 3,
                             'neutral': 0, 'symphaty': 5, 'sadness_despair': 6, 'sarcasm': 7}

        self.cat_dict = {'Notariswissels': 17, 'Wisseling betaaloptie': 1, 'Beheervragen': 2,
                         'Geen erfpachtrecht op naam': 3, 'Coulanceregeling': 4, 'AB1994': 5, 'Betalingsachterstand': 6,
                         'Foutieve bestemmingen': 7, 'TAG-vragen': 8, 'Annulering': 9, 'EPC': 10, '(ont)koppelen': 11,
                         'Doorverwijzen naar Overstapportaal': 12, 'Financieel': 13, 'Foutmeldingen': 14,
                         'Engelstalige vragen': 15, 'Vragen over brieven e.d.': 16, 'Status vragen': 0}

        self.intent_dict = {'commisive': 2, 'inform': 1, 'question': 0, 'directive': 3, 'complaint': 4}

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            cat = [self.cat_dict[self.cat[i]] for i in idx]
            intent = [self.intent_dict[self.intent[i]] for i in idx]
            emotion = [self.emotion[self.emotion[i]] for i in idx]
            lengths = [self.lengths[i] for i in idx]
            embeddings = torch.index_select(self.embeddings, 0, idx)

        cat = self.cat_dict[self.cat[idx]]
        intent = self.intent_dict[self.intent[idx]]
        emotion = self.emotion_dict[self.emotion[idx]]
        lengths = self.lengths[idx]
        embeddings = self.embeddings[idx]

        return embeddings, lengths, cat, intent, emotion, self.target_names
