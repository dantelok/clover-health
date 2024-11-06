import unittest
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# Import your models here
# from your_model_file import LinearSVM, RandomForest, NonlinearSVMRBF

class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic dataset for testing
        X, y = make_classification(n_samples=100, n_features=2, random_state=42)
        X = StandardScaler().fit_transform(X)
        cls.X_train = torch.tensor(X, dtype=torch.float32)
        cls.y_train = torch.tensor(y, dtype=torch.float32)

    def test_linear_svm_initialization(self):
        model = LinearSVM(input_dim=2)
        self.assertIsNotNone(model.linear, "Linear layer should be initialized")

    def test_linear_svm_training(self):
        model = LinearSVM(input_dim=2)
        model = train_svm(self.X_train, self.y_train, input_dim=2, epochs=10)

        with torch.no_grad():
            predictions = model(self.X_train)
        self.assertEqual(predictions.shape[0], self.X_train.shape[0],
                         "Number of predictions should match input samples")

    def test_linear_svm_predict(self):
        model = LinearSVM(input_dim=2)
        model = train_svm(self.X_train, self.y_train, input_dim=2, epochs=10)

        with torch.no_grad():
            predictions = (model(self.X_train) > 0).float()
        self.assertTrue(torch.all(predictions >= 0) and torch.all(predictions <= 1),
                        "Predictions should be binary (0 or 1)")

    def test_random_forest_initialization(self):
        model = RandomForest(num_trees=5, input_dim=2)
        self.assertEqual(len(model.trees), 5, "Random Forest should initialize 5 trees")

    def test_random_forest_predict(self):
        model = RandomForest(num_trees=5, input_dim=2)
        model.train(self.X_train, self.y_train, epochs=5)
        predictions = model.predict(self.X_train)
        self.assertEqual(len(predictions), self.X_train.shape[0], "Prediction length should match input samples")

    def test_nonlinear_svm_rbf_initialization(self):
        model = NonlinearSVMRBF(gamma=0.1, C=1.0, lr=0.001, max_iter=10)
        self.assertIsNotNone(model.gamma, "Gamma should be set")
        self.assertIsNotNone(model.C, "Regularization parameter C should be set")

    def test_nonlinear_svm_rbf_training(self):
        model = NonlinearSVMRBF(gamma=0.1, C=1.0, lr=0.001, max_iter=10)
        model.fit(self.X_train, self.y_train)

        # Check that alphas are within bounds
        self.assertTrue(torch.all(model.alpha >= 0) and torch.all(model.alpha <= model.C),
                        "Alpha values should be within [0, C]")

    def test_nonlinear_svm_rbf_predict(self):
        model = NonlinearSVMRBF(gamma=0.1, C=1.0, lr=0.001, max_iter=10)
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_train)
        self.assertEqual(len(predictions), self.X_train.shape[0], "Prediction length should match input samples")
        self.assertTrue(torch.all(predictions >= 0) and torch.all(predictions <= 1),
                        "Predictions should be binary (0 or 1)")


# Run the tests
if __name__ == '__main__':
    unittest.main()
