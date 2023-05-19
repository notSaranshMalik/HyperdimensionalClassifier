# HyperdimensionalClassifier

## Overview
A basic hyperdimensional vector classification algorithm made from scratch in Python, using Numpy to process the data.

## Using hyperdimensional vectors

### Vector creation
```
vector = Vector(size, zero_vec=False, no_init=False)
```
- `size` is an integer value, determining the dimensionality of the individual vector.
- `zero_vec` is an optional boolean value (false by default). If set to true, the vector created is not random, rather a zero-vector.
- `no_init` is an optional boolean value (false by default). If set to true, the vector created does not start with a backend array.

### Vector usage
- `v3 = v1 + v2`: Vector-vector addition works through bundling, with component-wise numeric addition.
- `v3 = v1 - v2`: Vector-vector subtraction works through bundling, with component-wise numeric subtraction.
- `v4 = v1 * v2`: Vector-vector multiplication works through binary XOR.
- `v2 = 3 * v1`: Scalar-vector multiplication works through component-wise multiplication. Note: right-multiplication of a scalar isn't valid due to speed optimisations.
- `v1 == v2`: Vector equality is overloaded with the conjunction of all component-wise comparisons.

### Vector functions
- `v_quant = vector.quantise()`: Vector quantisation to convert a bundled non-binary vector into a binary format. The midpoint of the minimum and maximum value is found, which is used as a division point. Note: the exact division point is added to the positive classification (i.e. quantised as a 1). The quantisation of a 2-vector sum has an expected n/4 bits at 1 if this isn't done, which causes an XOR multiplication to be too close to an original point.
- `sim = vector.hammingDistance(vector2)`: Finds the Hamming Distance between the two vectors. Useful for binary vectors.
- `sim = vector.cosineSimilarity(vector2)`: Finds the Cosine Similarity between the two vectors. Useful for integer vectors.
- `v_perm = vector.permute(perm, n)`: Vector perumtation using the perm variable. A given permutation can be generated with `np.random.permutation(size)` with `size` being the vector dimension. The permutation occurs iteratively, n times.
- `v_rev_perm = vector.permuteReverse(perm, n)`: Reverses the permutation n times.

### Vector groups
- `VectorGroups.blankVectors(n)` returns a list of n blank uninitialised vectors.
- `VectorGroups.zeroVectors(n)` returns a list of n zero vectors.
- `VectorGroups.randomVectors(n)` returns a list of n random vectors.
- `VectorGroups.levelVectors(n)` returns a list of n leveled vectors.

## Using a hyperdimensional vector space

### Vector space creation
```
space = VectorSpace(size, type="BIN")
```
- `size` is an integer value, determining the dimensionality of the individual vectors in this space.
- `type` is a string value, determining if the vector space is a "BIN" binary vector space (all vectors inserted should be quantised, and probing uses Hamming Distance) or an "INT" integer vector space (probing uses Cosine Similarity).

### Vector space functions
- `vector = space.newVector(label=None)`: Creates a new vector, then adds it to the vector space. If a label is given, it is attached to the vector. Acts as a shortcut method (see class notes below).
- `space.insertVector(vector, label=None)`: Inserts a given vector into the vector space. If a label is given, it is attached to the vector.
- `vec, perm = space.newSequence(*args, label=None)`: Inserts the given set of vectors into the vector space using permutation sums. Using a reverse permutation on the vector with the perm given, you can probe for the last element, twice for the second last, etc.
- `val = space.get(vector)`: Probes the vector space with the given vector, then returns either the label of the closest vector found (if the label was given), else the vector itself.

### Class notes
Create a vector using the vector class directly if it is meant for intermediary work and doesn't need to be searched. 
If it's required to be in the space, either create a vector `v = Vector(size)`, then insert it into the space `space.insertVector(v)`, or use the shortcut method to create it directly in the space `space.newVector()`.

## Using the classification algorithm

### Classifier creation
```
cl = Classifier(P=None, type="BIN")
```
- `P` is an optional integer value (os.cpu_count() - 1 by default) that determines how many cores the program will be run on at once. 
- `type` is an optional string value ("BIN" by default) that determines if the underlying vector space is binary (BIN) or integer (INT) based.

### Classification training
A simple training method
```
cl.train(X, y)
```
- `X` is a m by n Numpy matrix, where there are m datapoints and n features per point.
- `y` is a m long Numpy array, where there are m classifications for the m datapoints.

Uses adaptive OnlineHD for a slower, but more accurate retraining system (one-pass)
```
cl.retrain(X, y, parts)
```
- `X` is a m by n Numpy matrix, where there are m datapoints and n features per point.
- `y` is a m long Numpy array, where there are m classifications for the m datapoints.
- `parts` is an integer, which determines the number of retraining sections (20 recommended)

### Classification testing
```
y_pred = cl.classify(X_test)
```
- `X_test` is a r by n Numpy matrix, where there are r datapoints and n features per point.

## Using the feature picking class

### Explanation

The FeaturePicker class analyses the features which aren't useful in distinguishing between classes, and returns a list of features that should be used (as a boolean array).

### Usage of the class
```
feats = FeaturePicker.pickFeatures(X, y)
X_train = X_train[:, feats]
```
- `X` is the input X data. Use validation data to prevent overfitting.
- `y` is the input y data. Use validation data to prevent overfitting.
