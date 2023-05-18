import numpy as np
from vector_space import VectorSpace
from vector import Vector
from vector_groups import VectorGroups
from tqdm import tqdm

SIZE = 10000

class Classifier:

    # Globals for nearest neighbours
    vec_space = None

    # Globals for encoding data
    features = None
    values = None
    enc_zero = None

    def __init__(self):
        '''
        Classification initialisation method
        '''
        self.vec_space = VectorSpace(SIZE)

    def train(self, X, y, enc_zero=True, no_print=False):
        '''
        Train the model on the data X (mxn matrix) and classification 
        y (mx1 matrix). enc_zero is a metric that defines whether zero values
        should be encoded or not, which is useful on binary input data.
        '''

        # Create HD vectors for every feature and value
        if not no_print:
            print("\n\nBEGIN TRAINING")

        vecs = VectorGroups.random_vectors(X.shape[1])
        features = dict(zip(range(X.shape[1]), vecs))

        vecs = VectorGroups.level_vectors(max(X)-min(X)+1)
        values = dict(zip(range(min(X), max(X)+1), vecs))

        # Create an empty HD point for every class
        unique_y = np.unique(y)
        vecs = VectorGroups.zero_vectors(len(unique_y))
        classes = dict(zip(unique_y, vecs))

        classes = self._encode(X, y, features, values, classes, enc_zero)

        # Insert these class vectors into the vector space
        for cl, vec in classes.items():
            self.vec_space.insertVector(vec.quantise(), label=cl)

        # Save the features and values dicts to the instance class to use
        # during classification
        self.features = features
        self.values = values
        self.enc_zero = enc_zero

    def _encode(self, X, y, features, values, classes, enc_zero=True, t_pos=0):
        '''
        PRIVATE METHOD: DO NOT USE WITHOUT KNOWLEDGE
        Given X data, y data, the encoding data (features, values, enc_zero)
        and the classes to add into, this function encodes and returns all
        classifications
        '''
        for i in tqdm(range(X.shape[0]), position=t_pos, mininterval=0.5):
            sum = Vector(SIZE, zero_vec=True)
            for j in range(X.shape[1]):
                if X[i][j] == 0 and not enc_zero:
                    continue
                sum += features[j] * values[X[i][j]]
            classes[y[i]] += sum
        return classes

    def classify(self, X, t_pos=0, no_print=False):
        '''
        Classify a set of points X (mxn matrix)
        t_pos is an optional parameter for the line height of the progress bar
        in case multiple run at once
        '''

        # Classification begins
        if not no_print:
            print("\n\nBEGIN TESTING")
        y_hat = np.zeros(X.shape[0])
        for i in tqdm(range(X.shape[0]), position=t_pos, mininterval=0.5):
            sum = Vector(SIZE, zero_vec=True)
            for j in range(X.shape[1]):
                if X[i][j] == 0 and not self.enc_zero:
                    continue
                sum += self.features[j] * self.values[X[i][j]]
            y_hat[i] = self.vec_space.get(sum.quantise())
        
        return y_hat
