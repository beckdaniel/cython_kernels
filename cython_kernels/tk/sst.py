
import nltk
import cy_sst
import numpy as np
from collections import defaultdict

MAX_NODES = 10

class Node(object):
    """
    A node object.
    """
    def __init__(self, production, node_id, children_ids):
        self.production = production
        self.node_id = node_id
        self.children_ids = children_ids

    def __repr__(self):
        return str((self.production, self.node_id, self.children_ids))

class SubsetTreeKernel(object):
    """
    An SST Kernel, as defined by Moschitti, with
    two hyperparameters.
    """
    
    def __init__(self, _lambda=1, _sigma=1):
        self._lambda = _lambda
        self._sigma = _sigma
        self._tree_cache = {}

    def _gen_node_list(self, tree_repr):
        tree = nltk.Tree(tree_repr)
        c = 0
        node_list = []
        self._get_node(tree, node_list)
        node_list.sort(key=lambda x: x.production)
        node_dict = dict([(node.node_id, node) for node in node_list])
        return node_list, node_dict

    def _get_node(self, tree, node_list):
        if type(tree[0]) != str: #non preterm
            prod = [tree.node]
            children = []
            for ch in tree:
                ch_id = self._get_node(ch, node_list)
                prod.append(ch.node)
                children.append(ch_id)
            node_id = len(node_list)
            node = Node(' '.join(prod), node_id, children)
            node_list.append(node)
            return node_id
        else:
            prod = ' '.join([tree.node, tree[0]])
            node_id = len(node_list)
            node = Node(prod, node_id, None)
            node_list.append(node)
            return node_id            

    def _get_node_pairs(self, nodes1, nodes2):
        node_pair_list = []
        i1 = 0
        i2 = 0
        while True:
            try:
                if nodes1[i1].production > nodes2[i2].production:
                    i2 += 1
                elif nodes1[i1].production < nodes2[i2].production:
                    i1 += 1
                else:
                    while nodes1[i1].production == nodes2[i2].production:
                        reset2 = i2
                        while nodes1[i1].production == nodes2[i2].production:
                            node_pair_list.append((nodes1[i1], nodes2[i2]))
                            i2 += 1
                        i1 += 1
                        i2 = reset2
            except IndexError:
                break
        return node_pair_list

    def _get_node_pair_list_cy(self, nodes1, nodes2):
        return cy_sst.cy_get_node_pair_list(nodes1, nodes2)

    def _build_cache(self, X):
        #print X
        for tree_repr in X:
            t_repr = tree_repr[0]
            node_list, node_dict = self._gen_node_list(t_repr)
            self._tree_cache[t_repr] = (node_list, node_dict)
        
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
        #import ipdb
        #ipdb.set_trace()

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
                nodes1, dict1 = self._tree_cache[x1[0]]
                nodes2, dict2 = self._tree_cache[x2[0]]
                node_pairs = self._get_node_pairs(nodes1, nodes2)
                try:
                    K_result, ddecay_result = self.calc_K(node_pairs, dict1, dict2)
                    #import ipdb
                    #ipdb.set_trace()
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
            nodes, dicts = self._tree_cache[x[0]]
            node_pairs = self._get_node_pairs(nodes, nodes)
            delta_result = 0
            ddecay = 0
            # Calculation happens here.
            delta_result, ddecay = self.calc_K(node_pairs, dicts, dicts)
            K_vec[i] = delta_result
            ddecay_vec[i] = ddecay
        return (K_vec, ddecay_vec)

    def calc_K(self, node_pairs, dict1, dict2):
        K_total = 0
        ddecay_total = 0
        delta_matrix = np.zeros(shape=(MAX_NODES, MAX_NODES))
        for node_pair in node_pairs:
            K_result, ddecay_result = self.delta(node_pair[0], node_pair[1], delta_matrix, dict1, dict2)
            K_total += K_result
            ddecay_total += ddecay_result
        return (K_total, ddecay_total)

    def delta(self, node1, node2, delta_matrix, dict1, dict2):
        
        #CYTHON
        #########
        return cy_sst.cy_delta(node1, node2, delta_matrix, dict1, dict2, self._lambda, self._sigma)
        #########

        id1 = node1.node_id
        id2 = node2.node_id
        val = delta_matrix[id1, id2]
        if val > 0:
            return val, val
        if node1.children_ids == None:
            delta_matrix[id1, id2] = self._lambda
            return (self._lambda, 1)
        prod = 1
        for ch1, ch2 in zip(node1.children_ids, node2.children_ids):
            if dict1[ch1].production == dict2[ch2].production:
                K_result, ddecay_result = self.delta(dict1[ch1], dict2[ch2], 
                                                     delta_matrix, dict1, dict2)
                prod *= (self._sigma + K_result)
        result = self._lambda * prod
        delta_matrix[id1, id2] = result
        return result, result

