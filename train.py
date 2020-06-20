import json
import numpy as np
from nltk_utils import makeToken, stem, makeBag
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

inFile = open('intentions.json','r')
intents = json.load(inFile)

words = []
tags = []
combo = []

for i in intents['intents']:
    tag = tags.append(i["tag"])
    for w in i["patterns"]:
        token = makeToken(w)
        words.extend(token)
        combo.append((w, tag))

ignore = ['?','.',',','!']
words = [stem(w) for w in words if w not in ignore]
words = sorted(set(words))
tags = sorted(tags)

train_data1 = []
train_data2 = []
for (pattern, tag) in combo:
    bag = makeBag(pattern, words)
    train_data1.append(bag)

    label = tags.index(tag)
    train_data2.append(label)

np_data1 = np.array(train_data1)
np_data2 = np.array(train_data2)

class BotDataSet(Dataset):
    def __init__(self,np_data1,np_data2):
        self.numSamples = len(np_data1)
        self.data1 = np_data1
        self.data2 = np_data2
    
    def __getitem__(self,index):
        return (self.data1[index],self.data2[index])

    def __len__(self):
        return self.numSamples

batch_size = 8

dataset = BotDataSet(np_data1,np_data2)
training_info = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)