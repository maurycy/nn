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


class ActivationFunctions:
    @staticmethod
    def gelu(x):
        """Gaussian Error Linear Unit"""
        return (
            0.5
            * x
            * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
        )

    @staticmethod
    def swish(x):
        """Swish activation (x * sigmoid(x))"""
        return x * torch.sigmoid(x)


class HypertensorLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hyperoperation_order=2,
        activation="relu",
        use_residual=False,
    ):
        super(HypertensorLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hyperoperation_order = hyperoperation_order
        self.use_residual = use_residual

        # Activation function selection
        self.activation_name = activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = ActivationFunctions.gelu
        elif activation == "swish":
            self.activation = ActivationFunctions.swish
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Initialize weights
        scale = 0.01 / (2 ** (hyperoperation_order - 1))
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * scale)
        self.bias = nn.Parameter(torch.zeros(output_size))

        # Normalization layers
        self.input_bn = nn.BatchNorm1d(output_size)
        self.output_bn = nn.BatchNorm1d(output_size)

        # Residual projection if needed
        if use_residual and input_size != output_size:
            self.residual_proj = nn.Linear(input_size, output_size)
        else:
            self.residual_proj = None

        # Learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1) * scale)

    def forward(self, x):
        identity = x

        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        z = F.linear(x, self.weight, self.bias)
        z = self.input_bn(z)
        z = torch.tanh(z) * self.scale
        z = Hyperoperations.apply_hyperoperation(z, self.hyperoperation_order)
        z = self.output_bn(z)

        if self.use_residual:
            if self.residual_proj is not None:
                if len(identity.shape) > 2:
                    identity = identity.view(identity.size(0), -1)
                identity = self.residual_proj(identity)
            z = z + identity

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


class ConfigurableHypertensorNetwork(nn.Module):
    def __init__(
        self,
        input_shape=(28, 28),
        num_classes=10,
        hypertensor_config=[
            # {"size": 512, "order": 2, "activation": "gelu"},
            {"size": 1024, "order": 4, "activation": "gelu"},
        ],
        fc_layers=[1024, 1024],
        dropout_rate=0.3,
        use_residual=True,
    ):
        super(ConfigurableHypertensorNetwork, self).__init__()

        self.layers = nn.ModuleList()
        input_size = np.prod(input_shape)
        current_size = input_size

        # Build hypertensor layers
        for config in hypertensor_config:
            layer = HypertensorLayer(
                input_size=current_size,
                output_size=config["size"],
                hyperoperation_order=config["order"],
                activation=config.get("activation", "relu"),
                use_residual=use_residual,
            )
            self.layers.append(layer)
            current_size = config["size"]

        # Build FC layers
        self.fc_layers = nn.ModuleList()
        prev_size = current_size

        for fc_size in fc_layers:
            block = nn.ModuleList(
                [
                    nn.Linear(prev_size, fc_size),
                    nn.BatchNorm1d(fc_size),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            self.fc_layers.append(block)
            prev_size = fc_size

        self.output_layer = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        # Flatten input for first layer
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Pass through hypertensor layers
        for layer in self.layers:
            x = layer(x)

        # Pass through FC layers
        for block in self.fc_layers:
            for layer in block:
                x = layer(x)

        # Output layer
        x = self.output_layer(x)
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


def load_fashion_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST specific normalization
    ])

    train_dataset = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        "./data", train=False, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

# Helper function to get Fashion-MNIST class names
def get_fashion_mnist_labels():
    return [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

# Updated training function with class names
def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    class_names = get_fashion_mnist_labels()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.squeeze(1)  # Remove channel dimension
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

# Updated testing function with confusion matrix
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    class_names = get_fashion_mnist_labels()
    num_classes = len(class_names)
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.squeeze(1)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Update confusion matrix
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )

    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(num_classes):
        class_correct = confusion_matrix[i][i]
        class_total = confusion_matrix[i].sum()
        class_accuracy = 100.0 * class_correct / class_total
        print(f"{class_names[i]}: {class_accuracy:.2f}%")

    return test_loss, accuracy, confusion_matrix.cpu().numpy()

# Updated plotting function to include confusion matrix
def plot_results(run_name, train_losses, test_losses, train_accuracies, test_accuracies, confusion_matrix=None):
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Results for {run_name}")

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Test Loss")

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Test Accuracy")

    if confusion_matrix is not None:
        plt.subplot(1, 3, 3)
        class_names = get_fashion_mnist_labels()
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 15

    # Network configurations remain the same as before
    configs = [
        {
            "name": "RMSprop-Base",
            "hypertensor_config": [
                {"size": 256, "order": 3, "activation": "gelu"}
            ],
            "fc_layers": [32],
            "optimizer_config": {
                "lr": 0.001,
                "alpha": 0.99,
                "momentum": 0.0,
                "weight_decay": 0.0
            },
        },
        {
            "name": "RMSprop-Momentum",
            "hypertensor_config": [
                {"size": 256, "order": 3, "activation": "gelu"}
            ],
            "fc_layers": [64],
            "optimizer_config": {
                "lr": 0.001,
                "alpha": 0.99,
                "momentum": 0.9,
                "weight_decay": 0.0001
            }
        },
        {
            "name": "RMSprop-Scheduled",
            "hypertensor_config": [
                {"size": 128, "order": 2, "activation": "gelu"}
            ],
            "fc_layers": [64],
            "optimizer_config": {
                "lr": 0.002,
                "alpha": 0.95,
                "momentum": 0.9,
                "weight_decay": 0.0001
            },
            "scheduler_config": {
                "type": "OneCycleLR",
                "max_lr": 0.002,
                "pct_start": 0.3,
                "anneal_strategy": "cos"
            }
        }
    ]

    # Load Fashion-MNIST data instead of MNIST
    train_loader, test_loader = load_fashion_mnist_data(batch_size)

    for config in configs:
        print(f"\nTraining: {config['name']}")

        model = ConfigurableHypertensorNetwork(
            hypertensor_config=config['hypertensor_config'],
            fc_layers=config['fc_layers']
        ).to(device)

        optimizer = torch.optim.RMSprop(
            model.parameters(),
            **config['optimizer_config']
        )

        if 'scheduler_config' in config:
            scheduler_config = config['scheduler_config']
            if scheduler_config['type'] == 'OneCycleLR':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=scheduler_config['max_lr'],
                    epochs=epochs,
                    steps_per_epoch=len(train_loader),
                    pct_start=scheduler_config['pct_start'],
                    anneal_strategy=scheduler_config['anneal_strategy']
                )
                step_scheduler_batch = True
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2, verbose=True
                )
                step_scheduler_batch = False
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, verbose=True
            )
            step_scheduler_batch = False

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        learning_rates = []
        final_confusion_matrix = None

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(
                model, device, train_loader, optimizer, epoch,
                scheduler if step_scheduler_batch else None
            )

            test_loss, test_acc, confusion_matrix = test(model, device, test_loader)
            final_confusion_matrix = confusion_matrix  # Save the last epoch's confusion matrix

            if not step_scheduler_batch:
                scheduler.step(test_loss)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            learning_rates.append(optimizer.param_groups[0]['lr'])

            print(f'Epoch {epoch}:')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        # Plot results with confusion matrix
        plot_results(
            config["name"],
            train_losses,
            test_losses,
            train_accuracies,
            test_accuracies,
            final_confusion_matrix
        )

if __name__ == "__main__":
    main()

