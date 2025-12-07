import unittest
import numpy as np
from linear_regression import LinearRegression
from dataset_generator import generate_dataset

class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = generate_dataset(random_state=0)
        self.model = LinearRegression(learning_rate=0.01, n_iterations=1000)

    def test_fit(self):
        self.model.fit(self.X_train, self.y_train.ravel())
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)

    def test_predict(self):
        self.model.fit(self.X_train, self.y_train.ravel())
        predictions = self.model.predict(self.X_test)
        self.assertEqual(predictions.shape, self.y_test.ravel().shape)

    def test_score(self):
        self.model.fit(self.X_train, self.y_train.ravel())
        score = self.model.score(self.X_test, self.y_test.ravel())
        self.assertGreater(score, 0.5)

    def test_chaining(self):
        predictions = self.model.fit(self.X_train, self.y_train.ravel()).predict(self.X_test)
        self.assertEqual(predictions.shape, self.y_test.ravel().shape)

if __name__ == '__main__':
    unittest.main()


