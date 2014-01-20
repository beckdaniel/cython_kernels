
import nltk

class SubsetTreeKernel(object):
    """
    An SST Kernel, as defined by Moschitti, with
    two hyperparameters.
    """
    
    def __init__(self, _lambda=0.1, _sigma=1):
        self._lambda = _lambda
        self._sigma = _sigma
        self._tree_cache = {}

    def _gen_node_list(self, tree_repr):
        tree = nltk.Tree(tree_repr)
        pos = tree.treepositions()
        for l in tree.treepositions(order="leaves"):
            pos.remove(l)
        node_list = zip(tree.productions(), pos)
        node_list.sort()
        return node_list

    def _get_node_pair_list(self, nodes1, nodes2):
        node_pair_list = []
        i1 = 0
        i2 = 0
        while True:
            try:
                if nodes1[i1][0] > nodes2[i2][0]:
                    i2 += 1
                elif nodes1[i1][0] < nodes2[i2][0]:
                    i1 += 1
                else:
                    while nodes1[i1][0] == nodes2[i2][0]:
                        reset2 = i2
                        while nodes1[i1][0] == nodes2[i2][0]:
                            if type(nodes1[i1][0].rhs()[0]) == str:
                                # We consider preterms as leaves
                                tup = (nodes1[i1][1], nodes2[i2][1], 0)
                            else:
                                tup = (nodes1[i1][1], nodes2[i2][1], len(nodes1[i1][0].rhs()))
                            node_pair_list.append(tup)
                            i2 += 1
                        i1 += 1
                        i2 = reset2
            except IndexError:
                break
        node_pair_list.sort(key=lambda x: len(x[0]), reverse=True)
        return node_pair_list
        
