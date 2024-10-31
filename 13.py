import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from pathlib import Path


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
            {"size": 512, "order": 2, "activation": "gelu"},
            {"size": 64, "order": 3, "activation": "swish"},
        ],
        fc_layers=[512, 64],
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


# Enhanced version with improvements
class ImprovedConfigurableHypertensorNetwork(nn.Module):
    def __init__(
        self,
        input_shape=(28, 28),
        num_classes=10,
        hypertensor_config=[
            {"size": 256, "order": 2, "activation": "gelu"},  # Increased capacity
            {
                "size": 128,
                "order": 2,
                "activation": "swish",
            },  # Added second hypertensor layer
        ],
        fc_layers=[128, 64],
        dropout_rate=0.4,  # Increased dropout
        use_residual=True,
    ):
        super(ImprovedConfigurableHypertensorNetwork, self).__init__()

        # Input augmentation layer
        self.augment = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[0], input_shape[1])
            dummy = self.augment(dummy)
            flattened_size = self.flatten(dummy).shape[1]

        self.layers = nn.ModuleList()
        current_size = flattened_size

        # Build hypertensor layers with residual connections
        for config in hypertensor_config:
            layer = HypertensorLayer(
                input_size=current_size,
                output_size=config["size"],
                hyperoperation_order=config["order"],
                activation=config.get("activation", "gelu"),
                use_residual=use_residual,
            )
            self.layers.append(layer)
            current_size = config["size"]

        # Build FC layers with improved regularization
        self.fc_layers = nn.ModuleList()
        prev_size = current_size

        for fc_size in fc_layers:
            block = nn.ModuleList(
                [
                    nn.Linear(prev_size, fc_size),
                    nn.LayerNorm(fc_size),  # Changed to LayerNorm
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            self.fc_layers.append(block)
            prev_size = fc_size

        # Output layer with label smoothing
        self.output_layer = nn.Linear(prev_size, num_classes)

        # Initialize weights using better initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Apply input augmentation
        x = self.augment(x)
        x = self.flatten(x)

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


# Improved training function with mixup augmentation
def train_with_mixup(
    model, device, train_loader, optimizer, epoch, scheduler=None, alpha=0.2
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Apply mixup
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(data.size(0)).to(device)
            mixed_data = lam * data + (1 - lam) * data[index]

            optimizer.zero_grad()
            output = model(mixed_data)

            loss = lam * F.nll_loss(output, target) + (1 - lam) * F.nll_loss(
                output, target[index]
            )
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
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


def get_improved_training_config():
    return {
        "name": "Improved-Model",
        "model_config": {
            "hypertensor_config": [
                {"size": 256, "order": 2, "activation": "gelu"},
                {"size": 128, "order": 2, "activation": "swish"},
            ],
            "fc_layers": [128, 64],
            "dropout_rate": 0.4,
            "use_residual": True,
        },
        "optimizer_config": {"lr": 0.001, "weight_decay": 0.01, "betas": (0.9, 0.999)},
        "scheduler_config": {  # Removed 'type' parameter
            "max_lr": 0.003,
            "pct_start": 0.3,
            "div_factor": 10.0,
            "final_div_factor": 1000.0,
            "anneal_strategy": "cos",
        },
        "training_config": {"batch_size": 128, "epochs": 30, "mixup_alpha": 0.2},
    }


def load_fashion_mnist_data(batch_size=128, use_weighted_sampler=True):
    """
    Enhanced data loading with augmentation and weighted sampling
    """
    # Training augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    # Test transform without augmentation
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )

    # Load datasets
    train_dataset = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.FashionMNIST(
        "./data", train=False, transform=test_transform
    )

    # Create weighted sampler for training data
    if use_weighted_sampler:
        targets = train_dataset.targets
        class_counts = torch.bincount(targets)
        weights = 1.0 / class_counts.float()
        sample_weights = weights[targets]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    return train_loader, test_loader


def test_and_analyze(model, device, test_loader, epoch, save_dir=None):
    """
    Enhanced testing function with detailed analysis
    """
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    true_labels = []
    class_names = get_fashion_mnist_labels()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    # Generate classification report
    report = classification_report(
        true_labels, predictions, target_names=class_names, output_dict=True
    )

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Save results if directory is provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Confusion Matrix - Epoch {epoch}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_dir / f"confusion_matrix_epoch_{epoch}.png")
        plt.close()

        # Save classification report
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(save_dir / f"classification_report_epoch_{epoch}.csv")

    return test_loss, accuracy, report, cm


def train_one_epoch(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    scheduler=None,
    mixup_alpha=0.2,
    clip_grad_norm=1.0,
):
    """
    Enhanced training function for a single epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Apply mixup
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(data.size(0)).to(device)
            mixed_data = lam * data + (1 - lam) * data[index]
            mixed_target_a, mixed_target_b = target, target[index]

            optimizer.zero_grad()
            output = model(mixed_data)

            loss = lam * F.nll_loss(output, mixed_target_a) + (1 - lam) * F.nll_loss(
                output, mixed_target_b
            )
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)

        loss.backward()

        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        # Step the scheduler if it's batch-based
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

        if batch_idx % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Loss: {loss.item():.6f}\t'
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy


def main():
    # Set up configuration
    config = get_improved_training_config()

    # Set up device and random seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Create save directory
    save_dir = Path("fashion_mnist_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader, test_loader = load_fashion_mnist_data(
        batch_size=config["training_config"]["batch_size"], use_weighted_sampler=True
    )

    # Initialize model
    model = ImprovedConfigurableHypertensorNetwork(**config["model_config"]).to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer_config"]["lr"],
        weight_decay=config["optimizer_config"]["weight_decay"],
        betas=config["optimizer_config"]["betas"],
    )

    # Initialize scheduler - fixed initialization
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["scheduler_config"]["max_lr"],
        epochs=config["training_config"]["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=config["scheduler_config"]["pct_start"],
        div_factor=config["scheduler_config"]["div_factor"],
        final_div_factor=config["scheduler_config"]["final_div_factor"],
        anneal_strategy=config["scheduler_config"]["anneal_strategy"],
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "learning_rates": [],
    }

    best_test_acc = 0

    # Training loop
    for epoch in range(1, config["training_config"]["epochs"] + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            scheduler,
            config["training_config"]["mixup_alpha"],
        )

        # Test and analyze
        test_loss, test_acc, report, cm = test_and_analyze(
            model, device, test_loader, epoch, save_dir
        )

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "test_acc": test_acc,
                },
                save_dir / "best_model.pth",
            )

        # Print epoch results
        print(f"Epoch {epoch}:")
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Step epoch-based scheduler if used
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step(test_loss)

    # Plot final results
    plot_training_results(history, save_dir)

    return model, history


def plot_training_results(history, save_dir):
    """
    Plot and save training results
    """
    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Test Accuracy")
    plt.legend()

    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history["learning_rates"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "training_results.png")
    plt.close()


if __name__ == "__main__":
    main()
