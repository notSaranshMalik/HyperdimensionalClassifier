import numpy as np

class Vector:

    _vector = None

    def __init__(self, v_size, zero_vec = False):
        '''
        Creates a random hyperdimensional vector of size v_size
        If zero_vec, then the vector isn't randomised
        '''
        if not zero_vec:
            self._vector = np.random.randint(2, size=v_size)
        else:
            self._vector = np.zeros(v_size)

    def __mul__(self, other):
        '''
        Does vector multiplication through bitwise XOR
        Similar time compared to using sum and modulo
        '''
        ret = Vector(self._vector.size)
        ret._vector = 1*(self.quantise() != other.quantise())
        return ret
    __rmul__ = __mul__

    def __add__(self, other):
        '''
        Does vector addition through bundling
        '''
        ret = Vector(self._vector.size)
        ret._vector = self._vector + other._vector
        return ret
    __radd__ = __add__

    def __eq__(self, other):
        return (self.quantise() == other.quantise()).all()

    def quantise(self):
        '''
        Quantise a bundled vector into a binary vector
        '''
        boundary = (self._vector.max() - self._vector.min())/2 \
            + self._vector.min()
        return 1*(self._vector >= boundary)
    
    def permute(self, perm, n):
        '''
        Permute a vector based on a specific permutation, n times
        '''
        p = np.copy(self._vector)
        for _ in range(n):
            p = p[perm]
        ret = Vector(self._vector.size)
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
        ret = Vector(self._vector.size)
        ret._vector = p
        return ret
