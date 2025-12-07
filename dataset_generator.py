import numpy as np
from data_preprocessing import train_test_split

def generate_dataset(n_samples=200, n_features=1, noise=20, random_state=None):
    if random_state:
        np.random.seed(random_state)
        
    X = 2 * np.random.rand(n_samples, n_features)
    y = 2 + 9 * X + np.random.randn(n_samples, n_features) * noise
    
    return train_test_split(X, y, test_size=0.2, random_state=random_state)
