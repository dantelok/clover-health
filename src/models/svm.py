import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class NonlinearSVMRBF:
    def __init__(self, gamma=0.1, C=1.0, lr=0.001, max_iter=100):
        self.gamma = gamma  # Kernel coefficient
        self.C = C  # Regularization parameter
        self.lr = lr  # Learning rate
        self.max_iter = max_iter  # Number of iterations for training

    def rbf_kernel(self, X1, X2):
        """Compute the RBF (Gaussian) kernel between X1 and X2."""
        dist = torch.cdist(X1, X2) ** 2
        return torch.exp(-self.gamma * dist)

    def fit(self, X, y):
        # Convert labels to -1 and 1
        y = y.float() * 2 - 1

        # Store support vectors
        self.X_train = X
        self.y_train = y

        # Initialize alpha parameters for dual formulation
        self.alpha = nn.Parameter(torch.zeros(X.shape[0], requires_grad=True))

        # Define optimizer
        optimizer = optim.SGD([self.alpha], lr=self.lr)

        # Training loop
        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Calculate RBF kernel matrix
            K = self.rbf_kernel(X, X)

            # Dual SVM objective function
            loss = 0.5 * torch.sum((self.alpha * y) @ K @ (self.alpha * y)) - torch.sum(self.alpha)
            loss += self.C * torch.relu(1 - y * (K @ (self.alpha * y))).mean()  # Soft-margin penalty

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Enforce alpha constraints (for numerical stability)
            with torch.no_grad():
                self.alpha.clamp_(min=0, max=self.C)

            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Loss: {loss.item():.4f}")

    def decision_function(self, X):
        # Calculate kernel between X and support vectors
        K = self.rbf_kernel(X, self.X_train)
        return (K @ (self.alpha * self.y_train)).detach()

    def predict(self, X):
        # Classify based on the sign of the decision function
        decision_scores = self.decision_function(X)
        return (decision_scores > 0).float()

# Generate some example data
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Create a synthetic dataset for binary classification
X, y = make_classification(n_samples=200, n_features=2, random_state=42)
X = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Train the model
model = NonlinearSVMRBF(gamma=0.5, C=1.0, lr=0.001, max_iter=100)
model.fit(X, y)

# Predict on new data
predictions = model.predict(X)
print(predictions[:10])
