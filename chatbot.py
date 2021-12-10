import random
import json
import torch
from model import NeuralNetwork
from nltk_util import tokenize, bag_of_words

with open('data.json', 'r') as f:
    data = json.load(f)

FILE = "net.pth"
net = torch.load(FILE)

input_size = net["input_size"]
hidden_size = net["hidden_size"]
output_size = net["output_size"]
all_words = net["all_words"]
tags = net["tags"]
model_state = net["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

chatbot_name = "Charlie"
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, prediction = torch.max(output, dim = 1)
    tag = tags[prediction.item()]

    probs = torch.softmax(output, dim = 1)
    prob = probs[0][prediction.item()]

    if prob.item() > 0.75:
        for kw in data["data"]:
            if tag == kw["tag"]:
                print(f"{chatbot_name}: {random.choice(kw['responses'])}")

    else:
        print(f"{chatbot_name}: I do not understand...")