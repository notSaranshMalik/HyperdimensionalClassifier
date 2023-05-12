import numpy as np
from vector_space import VectorSpace
from vector import Vector
from tqdm import tqdm

SIZE = 10000

class Classifier:

    vec_space = None
    features = None
    values = None
    enc_zero = None

    def __init__(self):
        '''
        Classification initialisation method
        '''
        self.vec_space = VectorSpace(SIZE)

    def train(self, X, y, enc_zero=True):
        '''
        Train the model on the data X (mxn matrix) and classification 
        y (mx1 matrix). enc_zero is a metric that defines whether zero values
        should be encoded or not, which is useful on binary input data.
        '''

        # Create HD vectors for every feature and value
        features = dict()
        for i in range(X.shape[1]):
            features[i] = Vector(SIZE)

        values = dict()
        for i in np.unique(X):
            values[i] = Vector(SIZE)

        # Create an empty HD point for every class
        classes = dict()
        for i in np.unique(y):
            classes[i] = Vector(SIZE, zero_vec=True)

        # Create a binding for every training point
        for i in tqdm(range(X.shape[0])):
            sum = Vector(SIZE, zero_vec=True)
            for j in range(X.shape[1]):
                if X[i][j] == 0 and not enc_zero:
                    continue
                sum += features[j] * values[X[i][j]]
            classes[y[i]] += sum

        # Insert these class vectors into the vector space
        for cl, vec in classes.items():
            self.vec_space.insertVector(vec, label=cl)

        # Save the features and values dicts to the instance class to use
        # during classification
        self.features = features
        self.values = values
        self.enc_zero = enc_zero

    def classify(self, X):
        '''
        Classify a set of points X (mxn matrix)
        '''

        # Classification begins
        y_hat = np.zeros(X.shape[0])
        for i in tqdm(range(X.shape[0])):
            sum = Vector(SIZE, zero_vec=True)
            for j in range(X.shape[1]):
                if X[i][j] == 0 and not self.enc_zero:
                    continue
                sum += self.features[j] * self.values[X[i][j]]
            y_hat[i] = self.vec_space.get(sum)
        
        return y_hat
