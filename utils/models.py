import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output
import random


class TwoLayersNN(nn.Module):
    def __init__(self, width=512, input_size=28*28, output_bias=False):
        super(TwoLayersNN, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, width)  # First layer (784 inputs, width outputs)
        self.fc2 = nn.Linear(width, 1, bias=output_bias)  # Second layer (width inputs, 1 output)

        self.init_weights()

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = (x)
        return x

    def init_weights(self):
        # Initialize weights with small random values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

"""
def train(model, train_loader, test_loader, epochs=2000, lr=0.01, momentum=0.95):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        clear_output(wait=True)
        model.train()
        running_loss = sum(
            train_step(data, model, criterion, optimizer)
            for data in train_loader
        ) / len(train_loader)
        if (epoch + 1) % 200 == 0:
            clear_output(wait=True)
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {running_loss:.7f}')
            test_loss = evaluate(model, test_loader, criterion)
            print(f'Test Loss: {test_loss:.7f}')

    print('Finished Training')
    return test_loss, running_loss
"""

def train(model, train_loader, test_loader, epochs=2000, lr=0.01, momentum=0.95):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    alpha_list = []
    
    for epoch in range(epochs):
        clear_output(wait=True)
        model.train()
        initial_weights_first_layer = model.fc1.weight.data.clone()
        initial_weights_second_layer = model.fc2.weight.data.clone()
        running_loss = sum(
            train_step(data, model, criterion, optimizer)
            for data in train_loader
        ) / len(train_loader)
        final_weights_first_layer = model.fc1.weight.data
        final_weights_second_layer = model.fc2.weight.data

        delta_w1 = torch.mean(torch.abs(final_weights_first_layer - initial_weights_first_layer) / torch.abs(initial_weights_first_layer))
        delta_w2 = torch.mean(torch.abs(final_weights_second_layer - initial_weights_second_layer) / torch.abs(initial_weights_second_layer))

        # Calculate alpha and store it
        alpha = (delta_w1 / delta_w2).cpu().detach()
        if not(np.isnan(alpha) or np.isinf(alpha)):
            alpha_list.append(float(alpha.item()))
        
        if (epoch + 1) % 200 == 0:
            clear_output(wait=True)
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {running_loss:.7f}')
            test_loss = evaluate(model, test_loader, criterion)
            print(f'Test Loss: {test_loss:.7f}, alpha :{alpha}')

    print('Finished Training')
    return float(test_loss), float(running_loss), alpha_list

def train_step(data, model, criterion, optimizer):
    inputs, labels = (d.cuda() if torch.cuda.is_available() else d for d in data)
    #batch_size = inputs.size(0)
    optimizer.zero_grad()
    #inputs = inputs.reshape(batch_size, -1)
    outputs = model(inputs).flatten()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        losses = [
            criterion(model(inputs.cuda() if torch.cuda.is_available() else inputs).flatten(), 
                      labels.cuda() if torch.cuda.is_available() else labels).item()
            for inputs, labels in test_loader
        ]
    return sum(losses) / len(test_loader)