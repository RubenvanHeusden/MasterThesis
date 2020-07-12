import torch
from torch.utils.data import Dataset, DataLoader
import typing


class BertEmbeddingDailyDialogDataset(Dataset):
    def __init__(self, f_path: str = "../.data/dailydialog_bert/", mode="train"):
        self.target_names = ("emotion", "act", "topic")
        if mode == "train":
            self.embeddings, self.lengths, self.emotion, self.act, self.topic = torch.load(f_path+"train.pt")
        elif mode == "test":
            self.embeddings, self.lengths, self.emotion, self.act, self.topic = torch.load(f_path+"test.pt")
        else:
            raise(Exception("It seems that the selected dataloading mode is invalid"))

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            act = [self.act[i] for i in idx]
            topic = [self.topic[i] for i in idx]
            emotion = [self.emotion[i] for i in idx]
            lengths = [self.lengths[i] for i in idx]
            embeddings = torch.index_select(self.embeddings, 0, idx)

        act = self.act[idx]
        topic = self.topic[idx]
        emotion = self.emotion[idx]
        lengths = self.lengths[idx]
        embeddings = self.embeddings[idx]

        return embeddings, lengths, emotion, act, topic, self.target_names
