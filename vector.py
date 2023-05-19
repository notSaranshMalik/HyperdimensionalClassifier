import numpy as np
from numpy.linalg import norm

class Vector:

    _vector = None

    def __init__(self, v_size, zero_vec = False, no_init = False):
        '''
        Creates a random hyperdimensional vector of size v_size
        If zero_vec, then the vector isn't randomised
        If no_init, then the vector isn't initialised
        '''
        if no_init:
            return
        if not zero_vec:
            self._vector = np.random.randint(2, size=v_size)
        else:
            self._vector = np.zeros(v_size)

    def __mul__(self, other):
        '''
        Does vector multiplication through bitwise XOR
        Similar time compared to using sum and modulo
        '''
        ret = Vector(self._vector.size, no_init=True)
        ret._vector = 1*(self._vector != other._vector)
        return ret
    
    def __rmul__(self, other):
        '''
        Vector by scalar multiplcation
        '''
        ret = Vector(self._vector.size, no_init=True)
        ret._vector = self._vector * other
        return ret

    def __add__(self, other):
        '''
        Does vector addition through bundling
        '''
        ret = Vector(self._vector.size, no_init=True)
        ret._vector = self._vector + other._vector
        return ret

    def __sub__(self, other):
        '''
        Does vector subtraction through bundling
        '''
        ret = Vector(self._vector.size, no_init=True)
        ret._vector = self._vector - other._vector
        return ret

    def __eq__(self, other):
        '''
        Checks equality through component-wise comparisons
        '''
        return (self._vector == other._vector).all()
    
    def __str__(self):
        '''
        Returns a string representation of the underlying vector
        '''
        return self._vector.__str__()

    def quantise(self):
        '''
        Quantise a bundled vector into a binary vector
        '''
        boundary = (self._vector.max() - self._vector.min())/2 \
            + self._vector.min()
        ret = Vector(self._vector.size, no_init=True)
        ret._vector = 1*(self._vector >= boundary)
        return ret
    
    def hammingDistance(self, other):
        '''
        Calculates the Hamming Distance between two vectors
        '''
        return np.sum(other._vector != self._vector)

    def cosineSimilarity(self, other):
        '''
        Calculates the Cosine Similarity between two vectors
        '''
        a = self._vector
        b = other._vector
        return (a @ b) / (norm(a) * norm(b))
    
    def permute(self, perm, n):
        '''
        Permute a vector based on a specific permutation, n times
        '''
        p = np.copy(self._vector)
        for _ in range(n):
            p = p[perm]
        ret = Vector(self._vector.size, no_init=True)
        ret._vector = p
        return ret

    def permuteReverse(self, perm, n):
        '''
        Reverse a vector permutation based on the specific permutation, n times
        '''
        p = np.zeros(self._vector.size)
        cur = self._vector
        for _ in range(n):
            for i in range(self._vector.size):
                p[perm[i]] = cur[i]
            cur = np.copy(p)
        ret = Vector(self._vector.size, no_init=True)
        ret._vector = p
        return ret
