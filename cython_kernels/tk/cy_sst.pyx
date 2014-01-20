
from collections import defaultdict

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

def cy_delta(list node_list, double _lambda):
    cache_delta = defaultdict(int) # DP
    cache_ddecay = defaultdict(int)
    cdef list node1
    cdef list node2
    cdef unsigned int child_len
    cdef unsigned int i
    cdef double prod, sum_decay, ch_delta, ch_ddecay, delta_result
    cdef double delta_values, ddecay_values
    delta_values = 0
    ddecay_values = 0
    for node_pair in node_list:
        node1 = list(node_pair[0])
        node2 = list(node_pair[1])
        child_len = node_pair[2]
        #node1, node2, child_len = node_pair
        key = (node_pair[0], node_pair[1])
        if child_len == 0:
            cache_delta[key] = _lambda
            cache_ddecay[key] = 1
        else:
            prod = 1
            sum_decay = 0
            for i in xrange(child_len):
                child_key = (tuple(node1 + [i]),
                             tuple(node2 + [i]))
                ch_delta = cache_delta[child_key]
                ch_ddecay = cache_ddecay[child_key]
                prod *= 1 + ch_delta
                sum_decay += ch_ddecay / (1 + float(ch_delta))
            delta_result = _lambda * prod
            cache_delta[key] = delta_result
            delta_values += delta_result
            ddecay_result = prod + (delta_result * sum_decay)
            cache_ddecay[key] = ddecay_result
            ddecay_values += ddecay_result
    return (delta_values, ddecay_values)
