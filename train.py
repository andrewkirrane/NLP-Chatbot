import json
from nltk_util import tokenize, stem, bag_of_words
from nltk.corpus import stopwords
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork

with open('data.json', 'r') as f:
    data = json.load(f)

all_words = []
tags = []
pattern_tags = []

for data in data['data']:
    tag = data['tag']
    tags.append(tag)
    for pattern in data['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        pattern_tags.append((w, tag))

ignore_punct = ['?', '!', '.', ':', ';', ',']
all_words = [stem(w) for w in all_words if w in all_words if w not in ignore_punct or stopwords.words('english')]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (pattern_sentence, tag) in pattern_tags:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label) # Cross entropy loss

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)
learning_rate = 0.001
num_epochs = 2000

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

model = NeuralNetwork(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'Final loss, loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "net.pth"
torch.save(data,FILE)

print(f'Training complete, file saved to {FILE}')
