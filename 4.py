import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Correct import
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import wandb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import deque
import torchvision


class BaseHypernetwork(nn.Module):
    """Base class for all hypernetworks"""

    def __init__(self):
        super().__init__()
        self.monitor = None

    def init_monitoring(self, config=None):
        self.monitor = NetworkMonitor(self, config)

    def log_metrics(self, epoch, loss, outputs, targets):
        if self.monitor:
            self.monitor.log_metrics(epoch, loss, outputs, targets)


class StabilizedHyperoperations:
    """Base class for stable hyperoperations"""

    @staticmethod
    def power(base, exponent, max_val=100.0):
        base = torch.clamp(base, -10.0, 10.0)
        base = base + 1e-6  # Prevent zero gradients
        result = torch.pow(base, exponent)
        return torch.clamp(result, -max_val, max_val)

    @staticmethod
    def tetration(base, n, max_val=100.0):
        base = torch.clamp(base, -2.0, 2.0)
        result = base
        for _ in range(n - 1):
            if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
                return base
            result = StabilizedHyperoperations.power(base, result, max_val)
        return result


class StabilizedHypertensorLayer(nn.Module):
    """Stable implementation of hypertensor layer"""

    def __init__(self, input_shape, output_size, hyperoperation_order=2):
        super().__init__()
        self.input_shape = (
            input_shape if isinstance(input_shape, tuple) else (input_shape,)
        )
        self.output_size = output_size
        self.hyperoperation_order = hyperoperation_order

        # Initialize weights and biases
        weight_shape = (output_size,) + self.input_shape
        self.weight = nn.Parameter(torch.randn(weight_shape) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_size))

        # Normalization layers
        self.input_norm = nn.LayerNorm(self.input_shape)
        self.batch_norm = nn.BatchNorm1d(output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)

        # Expand dimensions for batch processing
        x_expanded = x.unsqueeze(1)
        weight_expanded = self.weight.unsqueeze(0)

        # Compute product
        hypertensor_product = x_expanded * weight_expanded

        # Sum across input dimensions
        sum_dims = tuple(range(2, 2 + len(self.input_shape)))
        z = hypertensor_product.sum(dim=sum_dims)

        # Add bias
        z = z + self.bias

        # Apply hyperoperation
        if self.hyperoperation_order == 2:
            z = StabilizedHyperoperations.power(z, 2)
        elif self.hyperoperation_order == 3:
            z = StabilizedHyperoperations.tetration(z, 2)
        else:
            raise ValueError("Only hyperoperations of order 2 and 3 are supported.")

        # Additional normalization and regularization
        z = self.batch_norm(z)
        z = self.dropout(z)

        return z


class NetworkMonitor:
    """Monitoring and logging utility"""

    def __init__(self, model, config, use_wandb=True):
        self.model = model
        self.writer = SummaryWriter("runs/hypernetwork_experiment")
        self.history = {
            "loss": deque(maxlen=1000),
            "gradients": deque(maxlen=1000),
            "weight_norms": deque(maxlen=1000),
            "activation_stats": deque(maxlen=1000),
            "stability_metrics": deque(maxlen=1000),
        }

        if use_wandb and config:
            wandb.init(project="hypernetwork-monitoring", config=config)

        self.activation_hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            if torch.isnan(output).any():
                print(f"NaN detected in {module.__class__.__name__}")
            stats = {
                "mean": output.mean().item(),
                "std": output.std().item(),
                "min": output.min().item(),
                "max": output.max().item(),
            }
            self.history["activation_stats"].append(stats)

        for name, module in self.model.named_modules():
            if isinstance(module, (StabilizedHypertensorLayer, nn.Linear)):
                self.activation_hooks.append(module.register_forward_hook(hook_fn))


class TextHypernetwork(BaseHypernetwork):
    """Hypernetwork for text processing"""

    def __init__(self, vocab_size, max_length=100, embedding_dim=100, num_classes=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        # Text embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Hypertensor processing
        self.hypertensor = StabilizedHypertensorLayer(
            input_shape=(max_length, embedding_dim),
            output_size=50,
            hyperoperation_order=2,
        )

        # Classification layers
        self.fc1 = nn.Linear(50, 20)
        self.fc2 = nn.Linear(20, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch_size, max_length)
        x = self.embedding(x)  # (batch_size, max_length, embedding_dim)
        x = self.hypertensor(x)  # (batch_size, 50)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ImageHypernetwork(BaseHypernetwork):
    """Hypernetwork for image processing"""

    def __init__(self, input_channels=3, image_size=(224, 224), num_classes=10):
        super().__init__()

        # Initial convolution layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate size after convolutions and pooling
        conv_output_size = (image_size[0] // 4, image_size[1] // 4)

        # Hypertensor layer
        self.hypertensor = StabilizedHypertensorLayer(
            input_shape=(64, conv_output_size[0], conv_output_size[1]),
            output_size=100,
            hyperoperation_order=2,
        )

        # Classification layers
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Convolutional feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Hypertensor processing
        x = self.hypertensor(x)

        # Classification
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class NumericalHypernetwork(BaseHypernetwork):
    """Hypernetwork for numerical data processing"""

    def __init__(self, input_features, output_size=1, hidden_size=50):
        super().__init__()

        self.input_norm = nn.BatchNorm1d(input_features)

        self.hypertensor = StabilizedHypertensorLayer(
            input_shape=(input_features,),
            output_size=hidden_size,
            hyperoperation_order=2,
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.hypertensor(x)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x


class DataProcessor:
    """Data preprocessing utility"""

    def __init__(self):
        self.text_tokenizer = None
        self.numerical_scaler = StandardScaler()

    def process_text(self, texts, max_length=100):
        """Process text data"""
        if self.text_tokenizer is None:
            # Simple tokenizer - in practice, use a proper tokenizer like BERT
            self.text_tokenizer = {
                word: idx + 1
                for idx, word in enumerate(
                    set(word for text in texts for word in text.split())
                )
            }

        processed = []
        for text in texts:
            tokens = [self.text_tokenizer.get(word, 0) for word in text.split()]
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
            processed.append(tokens)

        return torch.tensor(processed)

    def process_image(self, images, size=(224, 224)):
        """Process image data"""
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        processed = []
        for img in images:
            processed.append(transform(img))

        return torch.stack(processed)

    def process_numerical(self, data):
        """Process numerical data"""
        if isinstance(data, pd.DataFrame):
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            data[numerical_cols] = self.numerical_scaler.fit_transform(
                data[numerical_cols]
            )

            categorical_cols = data.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                dummies = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data, dummies], axis=1)
                data.drop(col, axis=1, inplace=True)

            return torch.tensor(data.values, dtype=torch.float32)
        else:
            return torch.tensor(
                self.numerical_scaler.fit_transform(data), dtype=torch.float32
            )


# Training utilities
class TrainingManager:
    """Manages the training process"""

    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Log metrics
            self.model.log_metrics(batch_idx, loss, output, target)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)

        if self.scheduler is not None:
            self.scheduler.step(avg_loss)

        return avg_loss


class ExampleUsage:
    """Examples of using hypernetworks with different data types"""

    @staticmethod
    def text_classification_example():
        # Example text classification
        texts = [
            "this movie was great",
            "terrible waste of time",
            "amazing performance",
            "very disappointing",
        ]
        labels = torch.tensor([1, 0, 1, 0])  # 1 for positive, 0 for negative

        # Initialize processor and process data
        processor = DataProcessor()
        processed_texts = processor.process_text(texts)

        # Create dataset
        dataset = torch.utils.data.TensorDataset(processed_texts, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

        # Initialize model
        vocab_size = len(processor.text_tokenizer) + 1
        model = TextHypernetwork(vocab_size=vocab_size, num_classes=2)

        # Initialize training components
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        # Initialize training manager
        trainer = TrainingManager(model, criterion, optimizer, scheduler)

        # Train model
        num_epochs = 100000
        for epoch in range(num_epochs):
            loss = trainer.train_epoch(dataloader)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    @staticmethod
    def image_classification_example():
        # Example image classification with MNIST
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Load MNIST dataset
        train_dataset = torchvision.datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            "./data", train=False, transform=transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

        # Initialize model
        model = ImageHypernetwork(input_channels=1, image_size=(28, 28), num_classes=10)

        # Initialize training components
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        # Initialize training manager
        trainer = TrainingManager(model, criterion, optimizer, scheduler)

        # Train model
        num_epochs = 10000
        for epoch in range(num_epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(test_loader)
            print(
                f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )


class AdvancedTraining:
    """Advanced training utilities and experiments"""

    def __init__(self, model_type, config):
        self.model_type = model_type
        self.config = config
        self.model = self._initialize_model()
        self.trainer = self._initialize_trainer()

    def _initialize_model(self):
        if self.model_type == "text":
            return TextHypernetwork(**self.config["model_params"])
        elif self.model_type == "image":
            return ImageHypernetwork(**self.config["model_params"])
        elif self.model_type == "numerical":
            return NumericalHypernetwork(**self.config["model_params"])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _initialize_trainer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["training_params"]["learning_rate"]
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config["training_params"]["lr_factor"],
            patience=self.config["training_params"]["patience"],
        )

        if self.model_type in ["text", "image"]:
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss()

        return TrainingManager(self.model, criterion, optimizer, scheduler)

    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.trainer.train_epoch(train_loader)

            # Validation phase
            val_loss = self.trainer.validate(val_loader)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    f"best_model_{self.model_type}.pth",
                )

            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("-" * 50)


class ExperimentRunner:
    """Runs experiments with different configurations"""

    @staticmethod
    def run_experiment(config):
        # Set random seeds for reproducibility
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

        # Initialize data processor
        processor = DataProcessor()

        # Process data based on type
        if config["data_type"] == "text":
            processed_data = processor.process_text(
                config["data"]["texts"], max_length=config["data_params"]["max_length"]
            )
        elif config["data_type"] == "image":
            processed_data = processor.process_image(
                config["data"]["images"], size=config["data_params"]["image_size"]
            )
        elif config["data_type"] == "numerical":
            processed_data = processor.process_numerical(config["data"]["features"])

        # Create dataset and loaders
        dataset = torch.utils.data.TensorDataset(
            processed_data, torch.tensor(config["data"]["labels"])
        )

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["training_params"]["batch_size"],
            shuffle=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config["training_params"]["batch_size"]
        )

        # Initialize and train model
        advanced_training = AdvancedTraining(
            model_type=config["data_type"], config=config
        )

        advanced_training.train(
            train_loader, val_loader, config["training_params"]["num_epochs"]
        )

        return advanced_training.model


def enhanced_text_classification_example():
    """Enhanced example with detailed monitoring"""
    # Example text data
    texts = [
        "this movie was great",
        "terrible waste of time",
        "amazing performance",
        "very disappointing",
        "excellent work",
        "poor quality",
        "outstanding result",
        "below average",
    ]
    # 1 for positive, 0 for negative
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])

    # Initialize processor and process data
    processor = DataProcessor()
    processed_texts = processor.process_text(texts)

    print(f"\nVocabulary size: {len(processor.text_tokenizer)}")
    print(f"Sample tokenized text: {processed_texts[0][:10]}...")

    # Create dataset
    dataset = torch.utils.data.TensorDataset(processed_texts, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model
    vocab_size = len(processor.text_tokenizer) + 1
    model = EnhancedTextHypernetwork(vocab_size=vocab_size, num_classes=2)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Initialize training components
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training loop with enhanced monitoring
    num_epochs = 10
    history = {"train_loss": [], "predictions": [], "layer_stats": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct = (pred == target).sum().item()
            epoch_correct += correct
            total_samples += len(target)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Store debug information
            if batch_idx == 0:  # Store first batch info
                history["layer_stats"].append(model.debug_info)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_correct / total_samples

        # Store metrics
        history["train_loss"].append(avg_loss)

        print(f"Epoch {epoch+1}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print("  Layer Statistics:")
        for layer_name, stats in model.debug_info.items():
            print(f"    {layer_name}:")
            print(f'      Mean: {stats["mean"]:.4f}')
            print(f'      Std: {stats["std"]:.4f}')
            print(f'      Shape: {stats["shape"]}')
        print()

        scheduler.step(avg_loss)

    # Final analysis
    print("\nTraining Summary:")
    print(f"Initial loss: {history['train_loss'][0]:.4f}")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(
        f"Loss improvement: {(history['train_loss'][0] - history['train_loss'][-1]):.4f}"
    )

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, history, processor


class EnhancedTextHypernetwork(BaseHypernetwork):
    """Enhanced version with better monitoring"""

    def __init__(self, vocab_size, max_length=100, embedding_dim=100, num_classes=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length

        # Text embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Hypertensor processing
        self.hypertensor = StabilizedHypertensorLayer(
            input_shape=(max_length, embedding_dim),
            output_size=50,
            hyperoperation_order=2,
        )

        self.fc1 = nn.Linear(50, 20)
        self.fc2 = nn.Linear(20, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Store intermediate values for monitoring
        self.debug_info = {}

        # Embedding
        embedded = self.embedding(x)
        self.debug_info["embedding"] = {
            "mean": embedded.mean().item(),
            "std": embedded.std().item(),
            "shape": embedded.shape,
        }

        # Hypertensor processing
        hypertensor_out = self.hypertensor(embedded)
        self.debug_info["hypertensor"] = {
            "mean": hypertensor_out.mean().item(),
            "std": hypertensor_out.std().item(),
            "shape": hypertensor_out.shape,
        }

        # Final layers
        x = F.relu(self.dropout(self.fc1(hypertensor_out)))
        self.debug_info["fc1"] = {
            "mean": x.mean().item(),
            "std": x.std().item(),
            "shape": x.shape,
        }

        x = self.fc2(x)
        logits = F.log_softmax(x, dim=1)
        self.debug_info["output"] = {
            "mean": logits.mean().item(),
            "std": logits.std().item(),
            "shape": logits.shape,
        }

        return logits


def enhanced_text_classification_example():
    """Enhanced example with detailed monitoring"""
    # Example text data
    texts = [
        "this movie was great",
        "terrible waste of time",
        "amazing performance",
        "very disappointing",
        "excellent work",
        "poor quality",
        "outstanding result",
        "below average",
    ]
    # 1 for positive, 0 for negative
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])

    # Initialize processor and process data
    processor = DataProcessor()
    processed_texts = processor.process_text(texts)

    print(f"\nVocabulary size: {len(processor.text_tokenizer)}")
    print(f"Sample tokenized text: {processed_texts[0][:10]}...")

    # Create dataset
    dataset = torch.utils.data.TensorDataset(processed_texts, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model
    vocab_size = len(processor.text_tokenizer) + 1
    model = EnhancedTextHypernetwork(vocab_size=vocab_size, num_classes=2)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Initialize training components
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training loop with enhanced monitoring
    num_epochs = 10
    history = {"train_loss": [], "predictions": [], "layer_stats": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct = (pred == target).sum().item()
            epoch_correct += correct
            total_samples += len(target)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Store debug information
            if batch_idx == 0:  # Store first batch info
                history["layer_stats"].append(model.debug_info)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_correct / total_samples

        # Store metrics
        history["train_loss"].append(avg_loss)

        print(f"Epoch {epoch+1}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print("  Layer Statistics:")
        for layer_name, stats in model.debug_info.items():
            print(f"    {layer_name}:")
            print(f'      Mean: {stats["mean"]:.4f}')
            print(f'      Std: {stats["std"]:.4f}')
            print(f'      Shape: {stats["shape"]}')
        print()

        scheduler.step(avg_loss)

    # Final analysis
    print("\nTraining Summary:")
    print(f"Initial loss: {history['train_loss'][0]:.4f}")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    print(
        f"Loss improvement: {(history['train_loss'][0] - history['train_loss'][-1]):.4f}"
    )

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, history, processor


# Run the enhanced example
if __name__ == "__main__":
    model, history, processor = enhanced_text_classification_example()

    # Test the model with new text
    def predict_sentiment(text, model, processor):
        model.eval()
        processed_text = processor.process_text([text])
        with torch.no_grad():
            output = model(processed_text)
            prob = torch.exp(output)[0]
            prediction = output.argmax(dim=1)
            return "Positive" if prediction.item() == 1 else "Negative", prob

    # Test cases
    test_texts = [
        "this was a fantastic movie",
        "what a terrible experience",
        "not bad at all",
        "could have been better",
    ]

    print("\nModel Predictions:")
    for text in test_texts:
        sentiment, prob = predict_sentiment(text, model, processor)
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment}")
        print(f"Confidence: Positive {prob[1]:.4f}, Negative {prob[0]:.4f}")
