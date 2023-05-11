import numpy as np
from classification import Classifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

'''
TEST 1: MNIST classification test
Using the 8x8 SKlearn sets
'''
def MNISTtest():

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)
    
    print("\n\nBEGIN TRAINING")
    MNIST_classifier = Classifier()
    MNIST_classifier.train(X_train, y_train)

    print("\n\nBEGIN TESTING")
    y_hat = MNIST_classifier.classify(X_test)

    print("\n\n")
    print(f"{X_train.shape[0]} training points")
    print(f"{X_test.shape[0]} testing points")
    print(f"{round(np.sum(y_hat == y_test) / y_hat.size, 2)} accuracy")
    print("\n\n")

'''
TEST 2: MNIST compressed classification test
Using a compressed form of the MNIST set where the values 0-16 are 
compressed to 0 and 1 - checking if the randomness of the hypervectors
gets mitigated through this
'''
def MNISTtestCompress(boundary):

    X, y = load_digits(return_X_y=True)
    X = 1 * (X >= boundary)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)
    
    print("\n\nBEGIN TRAINING")
    MNIST_classifier = Classifier()
    MNIST_classifier.train(X_train, y_train)

    print("\n\nBEGIN TESTING")
    y_hat = MNIST_classifier.classify(X_test)

    print("\n\n")
    print(f"{X_train.shape[0]} training points")
    print(f"{X_test.shape[0]} testing points")
    print(f"{round(np.sum(y_hat == y_test) / y_hat.size, 2)} accuracy")
    print("\n\n")

'''
Running tests
'''
if __name__ == "__main__":
    MNISTtest() # 2 runs - 0.72 and 0.79 accuracy at 70% testing
    MNISTtestCompress(8) # 2 runs - 0.76 and 0.85 accuracy at 70% testing
    MNISTtestCompress(1) # 2 runs - 0.84 and 0.84 accuracy at 70% testing
