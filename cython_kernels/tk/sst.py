
import nltk

class SubsetTreeKernel(object):
    """
    An SST Kernel, as defined by Moschitti, with
    two hyperparameters.
    """
    
    def __init__(self, _lambda=0.1, _sigma=1):
        self._lambda = _lambda
        self._sigma = _sigma

    def _gen_node_list(self, tree_repr):
        tree = nltk.Tree(tree_repr)
        pos = tree.treepositions()
        for l in tree.treepositions(order="leaves"):
            pos.remove(l)
        z = zip(tree.productions(), pos)
        z.sort()
        return z
        
        
