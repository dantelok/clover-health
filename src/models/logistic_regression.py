import torch
import torch.nn as nn
import torch.optim as optim

# Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # Define the linear layer
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Linear transformation followed by sigmoid activation
        return torch.sigmoid(self.linear(x))

# Example Usage
def train_logistic_regression(X_train, y_train, input_dim, learning_rate=0.01, epochs=100):
    # Initialize the model, loss function, and optimizer
    model = LogisticRegressionModel(input_dim)
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model

# Assuming you have data loaded as PyTorch tensors:
# X_train: Feature matrix (num_samples, input_dim)
# y_train: Labels (num_samples, 1)

# Example with dummy data
# Define input dimensions, create sample data
input_dim = 3  # Example: number of features in the dataset
X_train = torch.rand((100, input_dim))  # 100 samples, 3 features
y_train = torch.round(torch.rand(100, 1))  # 100 labels, 0 or 1

# Train the model
model = train_logistic_regression(X_train, y_train, input_dim, learning_rate=0.01, epochs=100)
