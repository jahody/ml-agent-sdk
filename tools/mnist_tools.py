"""
ML Tools for MNIST Dataset
Implements all machine learning operations for MNIST digit classification
"""

import numpy as np
import struct
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torchmetrics
import logging

# Configure logging to avoid excessive Lightning output
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

class MNISTLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(logits, y), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class SimpleCNN(MNISTLightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class ConfigurableMLP(MNISTLightningModule):
    def __init__(self, hidden_layers=[256, 128], learning_rate=0.001):
        super().__init__(learning_rate)
        layers = []
        input_dim = 784
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.model(x)

class SimpleResNet(MNISTLightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.res1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1)
        )
        self.fc = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        identity = x
        out = self.res1(x)
        out += identity
        out = F.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class MNISTMLTools:
    """Container for all MNIST ML tools used by the agent"""

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_flat = None
        self.X_test_flat = None
        self.model = None
        self.scaler = StandardScaler()
        self.image_shape = (28, 28)

    def _read_idx_images(self, filepath: str) -> np.ndarray:
        """Read IDX format image file"""
        with open(filepath, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows, cols)
        return images

    def _read_idx_labels(self, filepath: str) -> np.ndarray:
        """Read IDX format label file"""
        with open(filepath, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def load_data(self, train_images_path: str, train_labels_path: str,
                  test_images_path: str, test_labels_path: str) -> Dict[str, Any]:
        """Load MNIST training and test datasets"""
        try:
            self.X_train = self._read_idx_images(train_images_path)
            self.y_train = self._read_idx_labels(train_labels_path)
            self.X_test = self._read_idx_images(test_images_path)
            self.y_test = self._read_idx_labels(test_labels_path)

            return {
                "success": True,
                "train_shape": self.X_train.shape,
                "test_shape": self.X_test.shape,
                "train_samples": len(self.X_train),
                "test_samples": len(self.X_test),
                "image_dimensions": f"{self.X_train.shape[1]}x{self.X_train.shape[2]}",
                "num_classes": len(np.unique(self.y_train)),
                "train_label_distribution": {int(k): int(v) for k, v in 
                                            zip(*np.unique(self.y_train, return_counts=True))},
                "test_label_distribution": {int(k): int(v) for k, v in 
                                           zip(*np.unique(self.y_test, return_counts=True))},
                "pixel_value_range": f"[{self.X_train.min()}, {self.X_train.max()}]"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def explore_data(self, dataset: str = "train", num_samples: int = 5) -> Dict[str, Any]:
        """Explore MNIST dataset statistics"""
        X = self.X_train if dataset == "train" else self.X_test
        y = self.y_train if dataset == "train" else self.y_test

        if X is None or y is None:
            return {"success": False, "error": "Data not loaded"}

        try:
            # Get sample statistics
            sample_indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
            samples = []
            
            for idx in sample_indices:
                img = X[idx]
                samples.append({
                    "index": int(idx),
                    "label": int(y[idx]),
                    "mean_pixel_value": float(img.mean()),
                    "std_pixel_value": float(img.std()),
                    "non_zero_pixels": int(np.count_nonzero(img))
                })

            return {
                "success": True,
                "dataset": dataset,
                "total_samples": len(X),
                "image_shape": X.shape[1:],
                "label_distribution": {int(k): int(v) for k, v in 
                                      zip(*np.unique(y, return_counts=True))},
                "pixel_statistics": {
                    "mean": float(X.mean()),
                    "std": float(X.std()),
                    "min": int(X.min()),
                    "max": int(X.max())
                },
                "sample_images": samples
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def preprocess_data(self, operations: List[str]) -> Dict[str, Any]:
        """Preprocess MNIST data"""
        if self.X_train is None or self.X_test is None:
            return {"success": False, "error": "Data not loaded"}

        try:
            X_train = self.X_train.copy()
            X_test = self.X_test.copy()
            operations_applied = []

            # Normalize pixel values to [0, 1]
            if "normalize" in operations:
                X_train = X_train.astype('float32') / 255.0
                X_test = X_test.astype('float32') / 255.0
                operations_applied.append("normalized_to_0_1")

            # Standardize (zero mean, unit variance)
            if "standardize" in operations:
                X_train = (X_train.astype('float32') - 127.5) / 127.5
                X_test = (X_test.astype('float32') - 127.5) / 127.5
                operations_applied.append("standardized")

            # Flatten images for traditional ML models
            if "flatten" in operations:
                self.X_train_flat = X_train.reshape(X_train.shape[0], -1)
                self.X_test_flat = X_test.reshape(X_test.shape[0], -1)
                operations_applied.append("flattened_to_784")
            else:
                self.X_train_flat = X_train.reshape(X_train.shape[0], -1)
                self.X_test_flat = X_test.reshape(X_test.shape[0], -1)

            # Apply sklearn StandardScaler
            if "scale" in operations:
                self.X_train_flat = self.scaler.fit_transform(self.X_train_flat)
                self.X_test_flat = self.scaler.transform(self.X_test_flat)
                operations_applied.append("sklearn_scaled")

            return {
                "success": True,
                "operations_applied": operations_applied,
                "X_train_shape": self.X_train_flat.shape,
                "X_test_shape": self.X_test_flat.shape,
                "feature_count": self.X_train_flat.shape[1],
                "train_value_range": f"[{self.X_train_flat.min():.3f}, {self.X_train_flat.max():.3f}]",
                "test_value_range": f"[{self.X_test_flat.min():.3f}, {self.X_test_flat.max():.3f}]"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}



    def _train_pytorch_model(self, model_type: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Train a PyTorch Lightning model"""
        try:
            # Prepare data
            X_train_tensor = torch.FloatTensor(self.X_train_flat)
            y_train_tensor = torch.LongTensor(self.y_train)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            batch_size = hyperparameters.get('batch_size', 64)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model
            learning_rate = hyperparameters.get('learning_rate', 0.001)
            
            if model_type == 'cnn':
                self.model = SimpleCNN(learning_rate=learning_rate)
            elif model_type == 'pytorch_mlp':
                hidden_layers = hyperparameters.get('hidden_layers', [256, 128])
                self.model = ConfigurableMLP(hidden_layers=hidden_layers, learning_rate=learning_rate)
            elif model_type == 'resnet':
                self.model = SimpleResNet(learning_rate=learning_rate)
            
            # Trainer
            max_epochs = hyperparameters.get('max_epochs', 10)
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=1,
                enable_progress_bar=False,
                logger=False
            )
            
            # Train
            trainer.fit(self.model, dataloader)
            
            # Evaluate on training set
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X_train_tensor)
                preds = torch.argmax(logits, dim=1)
                train_accuracy = (preds == y_train_tensor).float().mean().item()
                
                # Evaluate on test set
                X_test_tensor = torch.FloatTensor(self.X_test_flat)
                y_test_tensor = torch.LongTensor(self.y_test)
                test_logits = self.model(X_test_tensor)
                test_preds = torch.argmax(test_logits, dim=1)
                test_accuracy = (test_preds == y_test_tensor).float().mean().item()
            
            return {
                "success": True,
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "training_samples": len(self.X_train_flat),
                "framework": "pytorch_lightning"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def train_model(self, model_type: str, hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train a machine learning model for digit classification"""
        if self.X_train_flat is None or self.y_train is None:
            return {"success": False, "error": "Data not preprocessed"}

        try:
            if hyperparameters is None:
                hyperparameters = {}
                
            # Handle PyTorch models
            if model_type in ['cnn', 'pytorch_mlp', 'resnet']:
                return self._train_pytorch_model(model_type, hyperparameters)

            # Initialize sklearn model
            if model_type == "random_forest":
                self.model = RandomForestClassifier(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', 20),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "logistic_regression":
                self.model = LogisticRegression(
                    max_iter=hyperparameters.get('max_iter', 1000),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "gradient_boosting":
                self.model = GradientBoostingClassifier(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', 5),
                    learning_rate=hyperparameters.get('learning_rate', 0.1),
                    random_state=42
                )
            elif model_type == "mlp":
                self.model = MLPClassifier(
                    hidden_layer_sizes=hyperparameters.get('hidden_layer_sizes', (128, 64)),
                    max_iter=hyperparameters.get('max_iter', 20),
                    random_state=42
                )
            else:
                return {"success": False, "error": f"Unknown model type: {model_type}"}

            # Train model
            self.model.fit(self.X_train_flat, self.y_train)

            # Training accuracy
            train_accuracy = self.model.score(self.X_train_flat, self.y_train)

            # Test accuracy (since we have labels)
            test_accuracy = self.model.score(self.X_test_flat, self.y_test)

            # Cross-validation on a subset (MNIST is large)
            subset_size = min(10000, len(self.X_train_flat))
            indices = np.random.choice(len(self.X_train_flat), subset_size, replace=False)
            cv_scores = cross_val_score(
                self.model, 
                self.X_train_flat[indices], 
                self.y_train[indices], 
                cv=3
            )

            return {
                "success": True,
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "cv_scores": cv_scores.tolist(),
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
                "training_samples": len(self.X_train_flat),
                "framework": "sklearn"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict(self, output_path: str = None) -> Dict[str, Any]:
        """Generate predictions on test set"""
        if self.model is None:
            return {"success": False, "error": "Model not trained"}

        if self.X_test_flat is None:
            return {"success": False, "error": "Test data not preprocessed"}

        try:
            # Generate predictions
            if isinstance(self.model, pl.LightningModule):
                self.model.eval()
                X_test_tensor = torch.FloatTensor(self.X_test_flat)
                with torch.no_grad():
                    logits = self.model(X_test_tensor)
                    predictions = torch.argmax(logits, dim=1).numpy()
            else:
                predictions = self.model.predict(self.X_test_flat)

            # Calculate accuracy if we have labels
            accuracy = None
            if self.y_test is not None:
                accuracy = float((predictions == self.y_test).mean())

            # Save predictions if path provided
            if output_path:
                np.savetxt(output_path, predictions, fmt='%d')

            # Get prediction distribution
            pred_distribution = {int(k): int(v) for k, v in 
                               zip(*np.unique(predictions, return_counts=True))}

            return {
                "success": True,
                "predictions_count": len(predictions),
                "predictions_sample": predictions[:20].tolist(),
                "prediction_distribution": pred_distribution,
                "test_accuracy": accuracy,
                "output_path": output_path
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
