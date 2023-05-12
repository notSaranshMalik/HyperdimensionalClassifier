import numpy as np
from classification import Classifier
'''
TEST 1: MNIST classification test
Using the 8x8 SKlearn set
'''
def MNISTTest():

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split


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
def MNISTTestCompress(boundary, enc_zero=True):

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    X = 1 * (X >= boundary)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)
    
    print("\n\nBEGIN TRAINING")
    MNIST_classifier = Classifier()
    MNIST_classifier.train(X_train, y_train, enc_zero=enc_zero)

    print("\n\nBEGIN TESTING")
    y_hat = MNIST_classifier.classify(X_test)

    print("\n\n")
    print(f"{X_train.shape[0]} training points")
    print(f"{X_test.shape[0]} testing points")
    print(f"{round(np.sum(y_hat == y_test) / y_hat.size, 2)} accuracy")
    print("\n\n")


'''
TEST 3: MNIST detailed classification test
Using the 28x28 TensorFlow set
'''
def MNISTTensortFlowTest(bound, train, test, enc_zero=True):

    from tensorflow import keras

    (X_train, y_train),(X_test, y_test) = \
        keras.datasets.mnist.load_data()
    
    X_train = 1*(X_train >= bound).reshape((X_train.shape[0], -1))[:train]
    y_train = y_train[:train]
    X_test = 1*(X_test >= bound).reshape((X_test.shape[0], -1))[:test]
    y_test = y_test[:test]
    
    print("\n\nBEGIN TRAINING")
    MNIST_classifier = Classifier()
    MNIST_classifier.train(X_train, y_train, enc_zero=enc_zero)

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
    # MNISTTest() # 4 runs - 0.76 average accuracy
    # MNISTTestCompress(8) # 2 runs - 0.81 accuracy
    # MNISTTestCompress(1) # 2 runs - 0.84 average accuracy
    # MNISTTestCompress(8, enc_zero=False) # 2 runs - 0.85 average accuracy
    # MNISTTestCompress(1, enc_zero=False) # 2 runs - 0.78 average accuracy
    MNISTTensortFlowTest(bound=30, train=10000, test=2000, enc_zero=False) # 3 runs - 0.70 average accuracy
    # MNISTTensortFlowTest(bound=128, train=10000, test=2000, enc_zero=False) # 1 run - 0.65 accuracy
    # MNISTTensortFlowTest(bound=85, train=60000, test=10000, enc_zero=False) # 1 run - 0.77 accuracy
