
import numpy as np
cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t

def cy_get_node_pair_list(list nodes1, list nodes2):
    cdef list node_pair_list = []
    cdef unsigned int i1 = 0
    cdef unsigned int i2 = 0
    cdef unsigned int reset2
    while True:
        try:
            prod1 = nodes1[i1][0]
            prod2 = nodes2[i2][0]
            if prod1 > prod2:
                i2 += 1
            elif prod1 < prod2:
                i1 += 1
            else:
                while prod1 == prod2:
                    reset2 = i2
                    while prod1 == prod2:
                        if type(prod1.rhs()[0]) == str:
                            # We consider preterms as leaves
                            tup = (nodes1[i1][1], nodes2[i2][1], 0)
                        else:
                            tup = (nodes1[i1][1], nodes2[i2][1], len(prod1.rhs()))
                        node_pair_list.append(tup)
                        i2 += 1
                        prod2 = nodes2[i2][0]
                    i1 += 1
                    i2 = reset2
                    prod1 = nodes1[i1][0]
                    prod2 = nodes2[i2][0]
        except IndexError:
            break
    node_pair_list.sort(key=lambda x: len(x[0]), reverse=True)
    return node_pair_list


def cy_delta(node1, node2, np.ndarray[DTYPE_t, ndim=2] delta_matrix, dict1, dict2, double _lambda, double _sigma):
    cdef unsigned int id1, id2, ch1, ch2, i
    #cdef int i
    cdef double val, prod, K_result, ddecay_result, result
    id1 = node1.node_id
    id2 = node2.node_id
    val = delta_matrix[id1, id2]
    if val > 0:
        return val, val
    if node1.children_ids == None:
        delta_matrix[id1, id2] = _lambda
        return (_lambda, 1)
    prod = 1
    children1 = node1.children_ids
    children2 = node2.children_ids
    for i in range(len(children1)):
        ch1 = children1[i]
        ch2 = children2[i]
        if dict1[ch1].production == dict2[ch2].production:
            K_result, ddecay_result = cy_delta(dict1[ch1], dict2[ch2], 
                                               delta_matrix, dict1, dict2, _lambda, _sigma)
            prod *= (_sigma + K_result)
    result = _lambda * prod
    delta_matrix[id1, id2] = result
    return result, result
