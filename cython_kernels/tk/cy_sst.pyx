

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
