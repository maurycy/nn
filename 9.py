import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class Hyperoperations:
    @staticmethod
    def stable_log1p(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def stable_exp(x):
        return torch.exp(torch.clamp(x, -10, 10))

    @staticmethod
    def power(base, exponent):
        return torch.pow(base, exponent)

    @staticmethod
    def tetration(base, height):
        base = torch.tanh(base) * 0.9
        result = base
        for _ in range(int(height) - 1):
            result = torch.pow(base, result)
            result = torch.tanh(result)
        return result

    @staticmethod
    def pentation(base, height):
        base = torch.tanh(base) * 0.5
        height = torch.clamp(height, 0, 3)
        if height <= 1:
            return base
        result = base
        for _ in range(int(height) - 1):
            log_result = Hyperoperations.stable_log1p(torch.abs(result))
            new_result = Hyperoperations.tetration(base, log_result)
            result = torch.tanh(new_result) * 0.5
        return result

    @staticmethod
    def apply_hyperoperation(x, order, n=2):
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

# HypertensorLayer remains mostly the same...
class HypertensorLayer(nn.Module):
    def __init__(
        self, input_size, output_size, hyperoperation_order=2, activation=None,
        use_residual=False
    ):
        super(HypertensorLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hyperoperation_order = hyperoperation_order
        self.activation = activation
        self.use_residual = use_residual

        # Initialize weights with scaled initialization
        scale = 0.01 / (2 ** (hyperoperation_order - 1))
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * scale)
        self.bias = nn.Parameter(torch.zeros(output_size))

        # Normalization layers
        self.input_bn = nn.BatchNorm1d(output_size)
        self.output_bn = nn.BatchNorm1d(output_size)

        # Residual projection if input and output sizes don't match
        if use_residual and input_size != output_size:
            self.residual_proj = nn.Linear(input_size, output_size)
        else:
            self.residual_proj = None

        # Learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1) * scale)

    def forward(self, x):
        # Store identity for residual connection
        identity = x

        # Reshape input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # Main computation
        z = F.linear(x, self.weight, self.bias)
        z = self.input_bn(z)
        z = torch.tanh(z) * self.scale
        z = Hyperoperations.apply_hyperoperation(z, self.hyperoperation_order)
        z = self.output_bn(z)

        # Apply residual connection if enabled
        if self.use_residual:
            if self.residual_proj is not None:
                if len(identity.shape) > 2:
                    identity = identity.view(identity.size(0), -1)
                identity = self.residual_proj(identity)
            z = z + identity

        if self.activation is not None:
            z = self.activation(z)

        return z

# Enhanced configurable network
class ConfigurableHypertensorNetwork(nn.Module):
    def __init__(
        self,
        input_shape=(28, 28),
        num_classes=10,
        hypertensor_config=[
            {"size": 128, "order": 2},
            {"size": 64, "order": 3}
        ],
        fc_layers=[128, 64],
        dropout_rate=0.3,
        use_residual=True
    ):
        super(ConfigurableHypertensorNetwork, self).__init__()

        self.layers = nn.ModuleList()

        # Calculate input size from input shape
        input_size = np.prod(input_shape)
        current_size = input_size

        # Build hypertensor layers
        for i, config in enumerate(hypertensor_config):
            layer = HypertensorLayer(
                input_size=current_size,
                output_size=config["size"],
                hyperoperation_order=config["order"],
                activation=F.relu,
                use_residual=use_residual
            )
            self.layers.append(layer)
            current_size = config["size"]

        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_size = current_size

        for fc_size in fc_layers:
            self.fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                nn.BatchNorm1d(fc_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = fc_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)

        # Adaptive dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Flatten input for first layer
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Pass through hypertensor layers
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        # Pass through FC layers
        for layer in self.fc_layers:
            x = layer(x)

        # Output
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

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


# Modified main function to use configurable network
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 10
    learning_rate = 0.001

    # Define different network configurations to try
    configs = [
        {
            "hypertensor_config": [
                {"size": 128, "order": 2}
            ],
            "fc_layers": [64],
            "name": "Single Hypertensor Layer"
        },
        {
            "hypertensor_config": [
                {"size": 128, "order": 2},
                {"size": 64, "order": 3}
            ],
            "fc_layers": [32],
            "name": "Double Hypertensor Layer"
        },
        {
            "hypertensor_config": [
                {"size": 128, "order": 2},
                {"size": 96, "order": 3},
                {"size": 64, "order": 2}
            ],
            "fc_layers": [32],
            "name": "Triple Hypertensor Layer"
        }
    ]

    train_loader, test_loader = load_mnist_data(batch_size)

    for config in configs:
        print(f"\nTraining network: {config['name']}")

        model = ConfigurableHypertensorNetwork(
            hypertensor_config=config['hypertensor_config'],
            fc_layers=config['fc_layers']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

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
        plt.suptitle(f"Results for {config['name']}")

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
