import scipy

import numpy as np

def nullspace(A, eps=1e-15):
    _, s, vh = scipy.linalg.svd(A)
    null_mask = s <= eps
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)


def spanning_vector(space):
    """ To obtain a spanning vector given a space of vectors, we take the sum of
        the vectors in the space
    """
    return sum(space)


def has_solution(eqn):
    # Track whether we've adjusted the denseness of eqn.A
    # so that we can reset it when we're done
    adjusted_denseness = False
    if not eqn.dense:
        eqn.dense = True
    nullspace_ = nullspace(eqn.A)
    if adjusted_denseness:
        eqn.dense = False
    if not nullspace_:
        return True
    nullspace_span = spanning_vector(nullspace_)
    has_solution_ = np.allclose(nullspace_span @ eqn.F, 0)
    return has_solution_

