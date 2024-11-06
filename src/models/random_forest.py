import numpy as np


class SimpleTree(nn.Module):
    def __init__(self, input_dim):
        super(SimpleTree, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class RandomForest:
    def __init__(self, num_trees, input_dim):
        self.trees = [SimpleTree(input_dim) for _ in range(num_trees)]

    def train(self, X_train, y_train, learning_rate=0.01, epochs=100):
        for i, tree in enumerate(self.trees):
            optimizer = optim.SGD(tree.parameters(), lr=learning_rate)
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = tree(X_train).squeeze()
                loss = nn.BCELoss()(output, y_train)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0 and i == 0:
                    print(f'Tree {i + 1}, Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X):
        tree_preds = [tree(X).detach().numpy() for tree in self.trees]
        return np.round(np.mean(tree_preds, axis=0))  # Majority vote across trees
