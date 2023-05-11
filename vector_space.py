import numpy as np
from vector import Vector

class VectorSpace:

    v_space = None
    v_size = None

    def __init__(self, v_size):
        '''
        Creates a hyperdimensional vector space of size v_size
        '''
        self.v_space = []
        self.v_size = v_size

    def newVector(self):
        '''
        Creates a new vector and adds it to the current vector space
        '''
        vec = Vector(self.v_size)
        self.v_space.append(vec)
        return vec
    
    def get(self, vec):
        '''
        Probe for the vector in the vector space
        '''
        size = len(self.v_space)
        dist = np.zeros(size)
        vec_quant = vec.quantise()
        for i in range(size):
            dist[i] = np.sum(vec_quant != self.v_space[i].quantise())
        return self.v_space[dist.argmin()]
    
    def newSequence(self, *v):
        '''
        Creates a new sequence using a multi permutation for each index 
        This is similar to ππππa + πππb + ππc + πd for a 4-sequence a, b, c, d
        '''
        perm = np.random.permutation(self.v_size)
        sum = Vector(self.v_size, zero_vec=True)
        for i in range(len(v)):
            sum += v[i].permute(perm, len(v)-i)
        return sum, perm