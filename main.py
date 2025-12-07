from dataset_generator import generate_dataset
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

def main():
    # Generate a dataset
    X_train, X_test, y_train, y_test = generate_dataset(random_state=42, noise=1)

    # Initialize and train the Linear Regression model
    model = LinearRegression(learning_rate=0.001, n_iterations=10000)
    model.fit(X_train, y_train.ravel())

    # Evaluate the model on the test set
    score = model.score(X_test, y_test.ravel())
    print(f"R-squared Score: {score}")

    # Make predictions
    predictions = model.predict(X_test)
    print(f"First 5 predictions: {predictions[:5]}")

    # Visualize the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, color='green', label='Test data')
    plt.plot(X_test, predictions, color='red', linewidth=2, label='Regression line')
    plt.xlabel('Feature (X)')
    plt.ylabel('Target (y)')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.savefig('output.png')
    plt.show()


if __name__ == "__main__":
    main()

