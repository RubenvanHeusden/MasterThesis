import os
import torch
from torch.utils.data import Dataset, DataLoader
import typing
import torch


class BertEmbeddingEnronDataset(Dataset):
    def __init__(self, f_path: str = "../.data/enron_bert/", mode="train"):
        self.target_names = ("category", "emotion")
        if mode == "train":
            self.embeddings, self.lengths, self.category, self.emotion = torch.load(f_path+"train.pt")
        elif mode == "test":
            self.embeddings, self.lengths, self.category, self.emotion = torch.load(f_path+"test.pt")
        else:
            raise(Exception("It seems that the selected dataloading mode is invalid"))

        self.emotion_dict = {0: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5,
                             9: 6, 10: 7, 11: 8, 12: 9}

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            category = [self.category[i] for i in idx]
            emotion = [self.emotion_dict[self.emotion[i]] for i in idx]
            lengths = [self.lengths[i] for i in idx]
            embeddings = torch.index_select(self.embeddings, 0, idx)

        category = self.category[idx]
        emotion = self.emotion_dict[self.emotion[idx]]
        lengths = self.lengths[idx]
        embeddings = self.embeddings[idx]

        return embeddings, lengths, category, emotion, self.target_names
