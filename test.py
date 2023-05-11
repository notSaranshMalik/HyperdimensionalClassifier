from vector_space import VectorSpace
import numpy as np

# Global setup
V_SIZE = 10000
HD_space = VectorSpace(V_SIZE)

'''
TEST 1: Kanerva language test
Testing the USA-Peso example from the Kanerva paper
'''
def test_language_once():
    COUNTRY_NAME = HD_space.newVector()
    usa = HD_space.newVector()
    mexico = HD_space.newVector()
    CURRENCY_NAME = HD_space.newVector()
    dollar = HD_space.newVector()
    peso = HD_space.newVector()

    usa_composite = COUNTRY_NAME * usa + CURRENCY_NAME * dollar
    mexico_composite = COUNTRY_NAME * mexico + CURRENCY_NAME * peso

    v_sum = 0
    v_sum += HD_space.get(usa_composite * dollar) == CURRENCY_NAME
    v_sum += HD_space.get(usa_composite * CURRENCY_NAME) == dollar
    v_sum += HD_space.get(usa_composite * usa) == COUNTRY_NAME
    v_sum += HD_space.get(usa_composite * COUNTRY_NAME) == usa
    v_sum += HD_space.get(mexico_composite * peso) == CURRENCY_NAME
    v_sum += HD_space.get(mexico_composite * CURRENCY_NAME) == peso
    v_sum += HD_space.get(mexico_composite * mexico) == COUNTRY_NAME
    v_sum += HD_space.get(mexico_composite * COUNTRY_NAME) == mexico
    return v_sum

def test_language():
    # Language test average - 100 times (8000 vectors by the last run)
    for i in range(100):
        val = test_language_once()
        print(f"Run {i+1}, Got {val}/8 calculations correct")

'''
TEST 2: Sequence dereferencing test
Testing the algorithm given in the Kanerva paper
'''
def test_sequence_once():
    item_a = HD_space.newVector()
    item_b = HD_space.newVector()
    item_c = HD_space.newVector()
    item_d = HD_space.newVector()

    seq, perm = HD_space.newSequence(item_a, item_b, item_c, item_d)
    
    v_sum = 0
    v_sum += HD_space.get(seq.permuteReverse(perm, 1)) == item_d
    v_sum += HD_space.get(seq.permuteReverse(perm, 2)) == item_c
    v_sum += HD_space.get(seq.permuteReverse(perm, 3)) == item_b
    v_sum += HD_space.get(seq.permuteReverse(perm, 4)) == item_a
    return v_sum

def test_sequence():
    # Sequence test average - 100 times (4000 vectors by the last run)
    for i in range(100):
        val = test_sequence_once()
        print(f"Run {i+1}, Got {val}/4 calculations correct")

'''
Running tests
'''
if __name__ == "__main__":
    test_language()
    test_sequence()