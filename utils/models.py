import numpy as np
import torch
import random


class TwoLayersNN(nn.Module):
    def __init__(self, width=512, input_size=28*28):
        super(Net, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, width)  # First layer (784 inputs, width outputs)
        self.fc2 = nn.Linear(width, 1, bias=False)  # Second layer (width inputs, 1 output)

        self.init_weights()

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def init_weights(self):
        # Initialize weights with small random values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

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
        print(f'Epoch {epoch + 1}/{epochs} - Loss: {running_loss:.3f}')

        test_loss = evaluate(model, test_loader, criterion)
        print(f'Test Loss: {test_loss:.3f}')

    print('Finished Training')
    return test_loss

def train_step(data, model, criterion, optimizer):
    inputs, labels = (d.cuda() if torch.cuda.is_available() else d for d in data)
    optimizer.zero_grad()
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