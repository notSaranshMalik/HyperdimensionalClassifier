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
Using the 8x8 SKlearn set - compressed onto the range 0-1
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
TEST 3: MNIST Tensor Flow classification test
Using the 28x28 TensorFlow set, compressed onto the range 0-1
'''
def MNISTTensorFlowTest(boundary, enc_zero=True):

    from tensorflow import keras

    (X_train, y_train),(X_test, y_test) = \
        keras.datasets.mnist.load_data()
    
    X_train = 1*(X_train >= boundary).reshape((X_train.shape[0], -1))
    y_train = y_train
    X_test = 1*(X_test >= boundary).reshape((X_test.shape[0], -1))
    y_test = y_test
    
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
    # MNISTTest() # 8 runs - 0.76 average accuracy
    # MNISTTestCompress(8) # 4 runs - 0.84 accuracy
    # MNISTTestCompress(8, enc_zero=False) # 2 runs - 0.85 average accuracy
    # MNISTTestCompress(1) # 2 runs - 0.84 average accuracy
    # MNISTTestCompress(1, enc_zero=False) # 2 runs - 0.78 average accuracy
    # MNISTTensorFlowTest(85, enc_zero=False) # 2 runs - 0.76 accuracy