
import nltk
import cy_sst
import numpy as np
from collections import defaultdict

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

    def _get_node_pair_list_cy(self, nodes1, nodes2):
        return cy_sst.cy_get_node_pair_list(nodes1, nodes2)

    def _build_cache(self, X):
        #print X
        for tree_repr in X:
            t_repr = tree_repr[0]
            node_list = self._gen_node_list(t_repr)
            self._tree_cache[t_repr] = node_list
        
    def K(self, X, X2, target):
        # A check to ensure that ddecays cache will always change when K changes
        self.ddecays = None
        if X2 == None:
            self.K_sym(X, target)
        else:
            self.K_nsym(X, X2, target)

    def K_sym(self, X, target):
        if self._tree_cache == {}:
            self._build_cache(X)

        # First, we are going to calculate K for diagonal values
        # because we will need them later to normalize.
        diag_deltas, diag_ddecays = self._diag_calculations(X)

        # Second, we are going to initialize the ddecay values
        # because we are going to calculate them at the same time as K.
        K_results = np.zeros(shape=(len(X), len(X)))
        ddecays = np.zeros(shape=(len(X), len(X)))
        
        # Now we proceed for the actual calculation
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X):
                if i > j:
                    K_results[i][j] = K_results[j][i]
                    ddecays[i][j] = ddecays[j][i]
                    continue
                if i == j:
                    K_results[i][j] = 1
                    continue
                # It will always be a 1-element array
                nodes1 = self._tree_cache[x1[0]]
                nodes2 = self._tree_cache[x2[0]]
                node_list = self._get_node_pair_list(nodes1, nodes2)
                try:
                    K_result, ddecay_result = self.delta(node_list)
                except:
                    print node_list
                    raise
                norm = diag_deltas[i] * diag_deltas[j]
                sqrt_norm = np.sqrt(norm)
                K_norm = K_result / sqrt_norm
                
                diff_term = ((diag_ddecays[i] * diag_deltas[j]) +
                             (diag_deltas[i] * diag_ddecays[j]))
                diff_term /= float(2 * norm)
                ddecay_norm = ((ddecay_result / sqrt_norm) -
                               (K_norm * diff_term))

                K_results[i][j] = K_norm
                ddecays[i][j] = ddecay_norm
        
        target += K_results
        self.ddecays = ddecays
        
    def _diag_calculations(self, X):
        K_vec = np.zeros(shape=(len(X),))
        ddecay_vec = np.zeros(shape=(len(X),))
        for i, x in enumerate(X):
            nodes = self._tree_cache[x[0]]
            node_list = self._get_node_pair_list(nodes, nodes)
            delta_result = 0
            ddecay = 0
            # Calculation happens here.
            delta_result, ddecay = self.delta(node_list)
            K_vec[i] = delta_result
            ddecay_vec[i] = ddecay
        return (K_vec, ddecay_vec)

    def delta(self, node_list):
        return cy_sst.cy_delta(node_list, self._lambda)

        cache_delta = defaultdict(int) # DP
        cache_ddecay = defaultdict(int)
        for node_pair in node_list:
            node1, node2, child_len = node_pair
            key = (node1, node2)
            if child_len == 0:
                cache_delta[key] = self._lambda
                cache_ddecay[key] = 1
            else:
                prod = 1
                sum_decay = 0
                for i in xrange(child_len):
                    child_key = (tuple(list(node1) + [i]),
                                 tuple(list(node2) + [i]))
                    ch_delta = cache_delta[child_key]
                    ch_ddecay = cache_ddecay[child_key]
                    prod *= 1 + ch_delta
                    sum_decay += ch_ddecay / (1 + float(ch_delta))
                delta_result = self._lambda * prod
                cache_delta[key] = delta_result
                cache_ddecay[key] = prod + (delta_result * sum_decay)
        return (sum(cache_delta.values()),
                sum(cache_ddecay.values()))

    def delta_cy(self, node_list):
        return cy_sst.cy_delta(node_list)
