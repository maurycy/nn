import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Enhanced Hyperoperations
# Enhanced Hyperoperations with better stabilization
class Hyperoperations:
    @staticmethod
    def stable_log1p(x):
        """Stabilized log1p that handles negative values"""
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def stable_exp(x):
        """Stabilized exp that prevents overflow"""
        return torch.exp(torch.clamp(x, -10, 10))

    @staticmethod
    def power(base, exponent):
        """Order 2 hyperoperation (power)"""
        return torch.pow(base, exponent)

    @staticmethod
    def tetration(base, height):
        """Order 3 hyperoperation (tetration) with improved stability"""
        # Scale inputs to prevent excessive growth
        base = torch.tanh(base) * 0.9  # Scale to [-0.9, 0.9]
        result = base

        for _ in range(int(height) - 1):
            result = torch.pow(base, result)
            # Apply tanh after each iteration to control growth
            result = torch.tanh(result)

        return result

    @staticmethod
    def pentation(base, height):
        """Order 4 hyperoperation (pentation) with logarithmic scaling"""
        # Scale inputs to very small range
        base = torch.tanh(base) * 0.5  # Scale to [-0.5, 0.5]
        height = torch.clamp(height, 0, 3)  # Limit height

        if height <= 1:
            return base

        result = base
        for _ in range(int(height) - 1):
            # Convert to log space
            log_result = Hyperoperations.stable_log1p(torch.abs(result))
            # Perform operation in log space
            new_result = Hyperoperations.tetration(base, log_result)
            # Scale result
            result = torch.tanh(new_result) * 0.5

        return result

    @staticmethod
    def apply_hyperoperation(x, order, n=2):
        """Apply hyperoperation of given order to tensor x with automatic scaling"""
        # Scale input based on hyperoperation order
        scale_factor = 1.0 / (2 ** (order - 1))
        x = x * scale_factor

        if order == 2:
            return Hyperoperations.power(x, n)
        elif order == 3:
            return Hyperoperations.tetration(x, n)
        elif order == 4:
            return Hyperoperations.pentation(x, n)
        else:
            raise ValueError("Hyperoperation order must be 2, 3, or 4")


# Improved Hypertensor Layer with adaptive scaling
class HypertensorLayer(nn.Module):
    def __init__(
        self, input_shape, output_size, hyperoperation_order=2, activation=None
    ):
        super(HypertensorLayer, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.hyperoperation_order = hyperoperation_order
        self.activation = activation

        # Adaptive initialization based on hyperoperation order
        weight_shape = (output_size,) + input_shape
        scale = 0.01 / (2 ** (hyperoperation_order - 1))
        self.weight = nn.Parameter(torch.randn(weight_shape) * scale)
        self.bias = nn.Parameter(torch.zeros(output_size))

        # Multiple batch norm layers for better stability
        self.input_bn = nn.BatchNorm1d(output_size)
        self.output_bn = nn.BatchNorm1d(output_size)

        # Learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1) * scale)

    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        weight_expanded = self.weight.unsqueeze(0)
        hypertensor_product = x_expanded * weight_expanded
        sum_dims = tuple(range(2, 2 + len(self.input_shape)))
        z = hypertensor_product.sum(dim=sum_dims)
        z = z + self.bias

        # Apply input batch norm
        z = self.input_bn(z)

        # Scale inputs adaptively
        z = torch.tanh(z) * self.scale

        # Apply hyperoperation with stabilization
        z = Hyperoperations.apply_hyperoperation(z, self.hyperoperation_order)

        # Apply output batch norm
        z = self.output_bn(z)

        if self.activation is not None:
            z = self.activation(z)

        return z


# Rest of the network remains the same...
class HypertensorMNISTNetwork(nn.Module):
    def __init__(self, input_shape=(28, 28), hidden_size=128, hyperoperation_order=2):
        super(HypertensorMNISTNetwork, self).__init__()
        self.hyperoperation_order = hyperoperation_order

        print(f"Initializing network with hyperoperation order {hyperoperation_order}")

        self.hypertensor_layer = HypertensorLayer(
            input_shape=input_shape,
            output_size=hidden_size,
            hyperoperation_order=hyperoperation_order,
            activation=F.relu,
        )

        # Adjusted hidden layer sizes based on hyperoperation order
        fc1_size = hidden_size // (hyperoperation_order - 1)

        self.fc1 = nn.Linear(hidden_size, fc1_size)
        self.bn1 = nn.BatchNorm1d(fc1_size)

        self.fc2 = nn.Linear(fc1_size, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 10)

        # Adaptive dropout based on hyperoperation order
        dropout_rate = 0.3 * (2 / hyperoperation_order)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.hypertensor_layer(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# Data loading and preprocessing
def load_mnist_data(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

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
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    accuracy = 100.0 * correct / total
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
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return test_loss, accuracy


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 10
    learning_rate = 0.001

    # Try different hyperoperation orders
    for order in [2, 3]:
        print(f"\nTraining with hyperoperation order {order}")

        # Load data
        train_loader, test_loader = load_mnist_data(batch_size)

        # Initialize model with specified hyperoperation order
        model = HypertensorMNISTNetwork(hyperoperation_order=order).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )

        # Training history
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        # Train and evaluate
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
            test_loss, test_acc = test(model, device, test_loader)
            scheduler.step(test_loss)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

        # Plot results
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Results for Hyperoperation Order {order}")

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Test Loss")

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(test_accuracies, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Training and Test Accuracy")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
