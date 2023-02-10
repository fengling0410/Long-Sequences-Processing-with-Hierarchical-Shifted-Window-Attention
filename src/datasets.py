import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

# create custom dataset class for arvix classification dataset
class ArvixDataset(Dataset):
    def __init__(self, path, tokenizer, model_config, mode='train', max_len=4096):

        self.dictCls2Idx = {
            "cs.AI": 0,
            "cs.cv": 1,
            "cs.IT": 2,
            "cs.PL": 3,
            "math.AC": 4,
            "math.ST": 5,
            "cs.CE": 6, 
            "cs.DS": 7,
            "cs.NE": 8,
            "cs.SY": 9 , 
            "math.GR": 10
        }
        self.Idx2dictCls = {}
        self.dataset = []
        self.labels  = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for sub in self.dictCls2Idx:
            label_index = self.dictCls2Idx[sub]
            subfolder = os.path.join(path,sub)
            self.Idx2dictCls[label_index] = sub

            files = sorted([f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder,f))])
            random.seed(1234)
            random.shuffle(files)

            if mode == "train":
                file_index = [i for i in range(model_config["train_size"])]
            elif mode == "validation":
                file_index = [i for i in range(model_config["train_size"], model_config["train_size"] + model_config["val_size"])]
            elif mode == "test":
                file_index = [i for i in range(model_config["train_size"] + model_config["val_size"], model_config["train_size"] + model_config["val_size"] + model_config["test_size"])]

            for i in file_index:
                f = files[i]
                fname = os.path.join(subfolder,f)
                self.dataset.append(fname)
                self.labels.append(label_index)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data_path = self.dataset[idx]
        data = self.read_txt(data_path)
        encoded_data = self.tokenizer.encode(data, truncation=True, padding="max_length", max_length=self.max_len)
        att_mask = torch.ones(len(encoded_data), dtype=torch.long)
        att_mask[0] = 2
        sample = {"Text": torch.tensor(encoded_data), 
                  "Attention": att_mask, 
                  "Label": torch.Tensor([label])}
        return sample

    def read_txt(self, file_path):
        with open(file_path, 'r') as file:
            text = file.read().replace('\n', '')
        return text