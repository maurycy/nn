import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperoperations (from your original code)
def hyperoperation_power(base, exponent):
    return torch.pow(base, exponent)

def hyperoperation_tetration(base, n):
    result = base
    for _ in range(n - 1):
        result = torch.pow(base, result)
    return result

# Hypertensor Layer (from your original code)
class HypertensorLayer(nn.Module):
    def __init__(self, input_shape, output_size, hyperoperation_order=2, activation=None):
        super(HypertensorLayer, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.hyperoperation_order = hyperoperation_order
        self.activation = activation

        weight_shape = (output_size,) + input_shape
        self.weight = nn.Parameter(torch.randn(weight_shape))
        self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        weight_expanded = self.weight.unsqueeze(0)
        hypertensor_product = x_expanded * weight_expanded
        sum_dims = tuple(range(2, 2 + len(self.input_shape)))
        z = hypertensor_product.sum(dim=sum_dims)
        z = z + self.bias

        if self.hyperoperation_order == 2:
            z = hyperoperation_power(z, 2)
        elif self.hyperoperation_order == 3:
            z = hyperoperation_tetration(z, 2)
        else:
            raise ValueError("Only hyperoperations of order 2 and 3 are supported.")

        if self.activation is not None:
            z = self.activation(z)
        return z

# Modified network for MNIST classification
class HypertensorMNISTNetwork(nn.Module):
    def __init__(self, input_shape=(28, 28), hidden_size=64):
        super(HypertensorMNISTNetwork, self).__init__()
        self.hypertensor_layer = HypertensorLayer(
            input_shape=input_shape,
            output_size=hidden_size,
            hyperoperation_order=2,
            activation=F.relu
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 10)  # 10 classes for MNIST
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.hypertensor_layer(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Data loading and preprocessing
def load_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.squeeze(1)  # Remove channel dimension
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.squeeze(1)  # Remove channel dimension
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128  # Increased batch size
    epochs = 20  # More epochs
    learning_rate = 0.0003  # Lower learning rate

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model and optimizer
    model = HypertensorMNISTNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Training history
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # Train and evaluate
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Test Accuracy')

    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()
