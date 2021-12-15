# Import modules and libraries
import random
import json
import torch
from model import NeuralNetwork
from nltk_funcs import tokenize, bag_of_words

# Open data file, save data to dictionary
with open('data.json', 'r') as f:
    data = json.load(f)

# Open and load trained model
FILE = "net.pth"
net = torch.load(FILE)

# Save trained model parameters
input_size = net["input_size"]
hidden_size = net["hidden_size"]
output_size = net["output_size"]
all_words = net["all_words"]
tags = net["tags"]
model_state = net["model_state"]

# Initialize new model based on trained model
model = NeuralNetwork(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# User input to name chatbot
chatbot_name = input("What would you like to name the chatbot?\n")

# Loop for conversation with chatbot
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input('You: ') # User input for next line in conversation
    # Escape from conversation
    if sentence == "quit":
        break

    sentence = tokenize(sentence) # Tokenize user input
    x = bag_of_words(sentence, all_words) # Create a bag of words with user input
    x = x.reshape(1, x.shape[0]) # Reshape to one row
    x = torch.from_numpy(x) # Convert numpy array to torch tensor

    output = model(x) # Feed user input data through model
    _, prediction = torch.max(output, dim = 1) # Returns prediction 
    tag = tags[prediction.item()] # Return predicted tag

    # Calculate probability of output being correct
    probs = torch.softmax(output, dim = 1)
    prob = probs[0][prediction.item()] 

    if prob.item() > 0.75: # Threshold of 75%, else bot does not understand
        # For tag in data dictionary
        for keyword in data["data"]:
            # If tag matches predicted tag
            if tag == keyword["tag"]:
                print(f"{chatbot_name}: {random.choice(keyword['responses'])}") # Print a random response from the list of responses

    else:
        print(f"{chatbot_name}: I do not understand...") # If under 75% probablility, print "I do not understand..."