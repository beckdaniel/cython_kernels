

import unittest
import nltk
from cython_kernels.tk.sst import SubsetTreeKernel as SST
import datetime
import numpy as np

class SSTTests(unittest.TestCase):

    def test_gen_node_list(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1, dict1 = k._gen_node_list(repr1)
        result = "[('ADJ colorless', 0, None), ('ADV furiously', 4, None), ('N ideas', 1, None), ('NP ADJ N', 2, [0, 1]), ('S NP VP', 6, [2, 5]), ('V sleep', 3, None), ('VP V ADV', 5, [3, 4])]"
        print nodes1
        self.assertEqual(str(nodes1), result)

    @unittest.skip("skip")
    def test_get_node_pairs1(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1, dict1 = k._gen_node_list(repr1)
        nodes2, dict2 = k._gen_node_list(repr2)
        node_list = k._get_node_pairs(nodes1, nodes2)
        result = "[(('ADJ colorless', 0, None), ('ADJ colorless', 0, None)), (('N ideas', 1, None), ('N ideas', 1, None)), (('NP ADJ N', 2, [0, 1]), ('NP ADJ N', 2, [0, 1])), (('V sleep', 3, None), ('V sleep', 3, None)), (('VP V ADV', 5, [3, 4]), ('VP V ADV', 5, [3, 4]))]"
        self.assertEqual(str(node_list), result)

    @unittest.skip("cy")
    def test_get_node_pair_list_cy(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1 = k._gen_node_list(repr1)
        nodes2 = k._gen_node_list(repr2)
        node_list = k._get_node_pair_list_cy(nodes1, nodes2)
        result = "[((0, 0), (0, 0), 0), ((1, 1), (1, 1), 0), ((0, 1), (0, 1), 0), ((1, 0), (1, 0), 0), ((0,), (0,), 2), ((1,), (1,), 2), ((), (), 2)]"
        self.assertEqual(str(node_list), result)

    def test_get_node_pairs2(self):
        repr1 = '(S (NP ns) (VP v))'
        repr2 = '(S (NP (N a)) (VP (V c)))'
        k = SST()
        nodes1, dict1 = k._gen_node_list(repr1)
        nodes2, dict2 = k._gen_node_list(repr2)
        node_list = k._get_node_pairs(nodes1, nodes2)
        print node_list
        print dict1
        print dict2

    def test_K(self):
        X = np.array([['(S (NP ns) (VP v))'],
                      ['(S (NP n) (VP v))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        k = SST()
        target = np.zeros(shape=(len(X), len(X)))
        k.K(X, None, target)
        result = [[ 1.,          0.5,         0.10540926,  0.08333333,  0.06711561],
                  [ 0.5,         1.,          0.10540926,  0.08333333,  0.06711561],
                  [ 0.10540926,  0.10540926,  1.,          0.31622777,  0.04244764],
                  [ 0.08333333,  0.08333333,  0.31622777,  1.,          0.0335578 ],
                  [ 0.06711561,  0.06711561,  0.04244764,  0.0335578,   1.        ]]
        self.assertAlmostEqual(np.sum(result), np.sum(target))



class SSTProfilingTests(unittest.TestCase):

    @unittest.skip("skip")
    def test_prof_gen_node_pair_list(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1 = k._gen_node_list(repr1)
        nodes2 = k._gen_node_list(repr2)
        start_time = datetime.datetime.now()
        for i in range(20000):
            node_list = k._get_node_pair_list(nodes1, nodes2)
        end_time = datetime.datetime.now()
        print end_time - start_time

    @unittest.skip("skip")
    def test_prof_gen_node_pair_list_cy(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1 = k._gen_node_list(repr1)
        nodes2 = k._gen_node_list(repr2)
        start_time = datetime.datetime.now()
        for i in range(20000):
            node_list = k._get_node_pair_list_cy(nodes1, nodes2)
        end_time = datetime.datetime.now()
        print end_time - start_time

    #@unittest.skip("skip")
    def test_prof_K(self):
        X = np.array([['(S (NP ns) (VP v))'],
                      ['(S (NP n) (VP v))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        k = SST()
        target = np.zeros(shape=(len(X), len(X)))
        ITS = 1000
        start_time = datetime.datetime.now()
        for i in range(ITS):
            k.K(X, None, target)
        end_time = datetime.datetime.now()
        print target/ITS
        print end_time - start_time

if __name__ == "__main__":
    unittest.main()
