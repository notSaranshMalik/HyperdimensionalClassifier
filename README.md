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
- `v3 = v1 + v2`: Vector addition works through bundling, with component-wise numeric addition. For speed reasons, all functions in this library assume quantised vectors, so always quantise after adding to prevent errors!
- `v4 = v1 * v2`: Vector multiplication works through binary XOR.
- `v1 == v2`: Vector equality is overloaded with the conjunction of all component-wise comparisons.

### Vector functions
- `v_quant = vector.quantise()`: Vector quantisation to convert a bundled non-binary vector into a binary format. The midpoint of the minimum and maximum value is found, which is used as a division point. Note: the exact division point is added to the positive classification (i.e. quantised as a 1). The quantisation of a 2-vector sum has an expected n/4 bits at 1 if this isn't done, which causes an XOR multiplication to be too close to an original point.
- `v_perm = vector.permute(perm, n)`: Vector perumtation using the perm variable. A given permutation can be generated with `np.random.permutation(size)` with `size` being the vector dimension. The permutation occurs iteratively, n times.
- `v_rev_perm = vector.permuteReverse(perm, n)`: Reverses the permutation n times.

## Using a hyperdimensional vector space

### Vector space creation
```
space = VectorSpace(size)
```
- `size` is an integer value, determining the dimensionality of the individual vectors in this space.

### Vector space functions
- `vector = space.newVector(label=None)`: Creates a new vector, then adds it to the vector space. If a label is given, it is attached to the vector. Acts as a shortcut method (see class notes below).
- `space.insertVector(vector, label=None)`: Inserts a given vector into the vector space. If a label is given, it is attached to the vector.
- `vec, perm = space.newSequence(*args, label=None)`: Inserts the given set of vectors into the vector space using permutation sums. Using a reverse permutation on the vector with the perm given, you can probe for the last element, twice for the second last, etc.
- `val = space.get(vector)`: Probes the vector space with the given vector, then returns either the label of the closest vector found (if the label was given), else the vector itself.

### Class notes
Create a vector using the vector class directly if it is meant for intermediary work and doesn't need to be searched. 
If it's required to be in the space, either create a vector `v = Vector(size)`, then insert it into the space `space.insertVector(v)`, or use the shortcut method to create it directly in the space `space.newVector()`.

## Using the classification algorithm

### Classifier creation and training
```
cl = Classifier()
cl.train(X, y, enc_zero=True)
```
- `X` is a m by n Numpy matrix, where there are m datapoints and n features per point.
- `y` is a m long Numpy array, where there are m classifications for the m datapoints.
- `enc_zero` is an optional boolean value (true by defauly). If set to false, the zero points of data on the input aren't encoded at all, which is useful for inputs with only boolean value features.

### Classification testing
```
y_pred = cl.classify(X_test)
```
- `X_test` is a r by n Numpy matrix, where there are r datapoints and n features per point.

## Example usage of vector space

### Example 1 - Kanerva's USD-Peso example
```
COUNTRY_NAME = HD_space.newVector()
usa = HD_space.newVector()
CURRENCY_NAME = HD_space.newVector()
dollar = HD_space.newVector()
usa_composite = COUNTRY_NAME * usa + CURRENCY_NAME * dollar

HD_space.get(usa_composite * dollar) == CURRENCY_NAME # True
```

### Example 2 - Sequencing
```
item_a = HD_space.newVector()
item_b = HD_space.newVector()
item_c = HD_space.newVector()
item_d = HD_space.newVector()

seq, perm = HD_space.newSequence(item_a, item_b, item_c, item_d)

HD_space.get(seq.permuteReverse(perm, 2)) == item_c # True
```

## Example usage of classification algorithm
```
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np 

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

cl = Classifier()
cl.train(X_train, y_train)
y_pred = cl.classify(X_test)

print(f"{round(np.sum(y_test == y_pred) / y_test.size, 2)} accuracy") # Typically 75% or more accuracy!
```

## Accuracy on classification datasets
1) 85% accuracy on 8x8 sklearn dataset (boundary 8, positive only encoded)
2) 76% accuracy on 28x28 TensorFlow dataset (boundary 85, positive only encoded)
