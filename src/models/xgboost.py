class XGBoostApprox:
    def __init__(self, num_rounds, input_dim, learning_rate=0.1):
        self.models = []
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        for _ in range(num_rounds):
            self.models.append(SimpleTree(input_dim))

    def train(self, X_train, y_train, epochs=50):
        residuals = y_train.clone()  # Start with original target

        for i, model in enumerate(self.models):
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = model(X_train).squeeze()
                loss = nn.MSELoss()(output, residuals)  # Minimize residuals
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                # Update residuals
                residuals -= self.learning_rate * model(X_train).squeeze()

    def predict(self, X):
        predictions = torch.zeros(X.shape[0], 1)
        for model in self.models:
            predictions += self.learning_rate * model(X)
        return torch.sigmoid(predictions).round()  # Sigmoid and round to get binary output
