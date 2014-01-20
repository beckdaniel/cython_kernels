

import unittest
import nltk
from cython_kernels.tk.sst import SubsetTreeKernel as SST
import datetime
import numpy as np

class SSTTests(unittest.TestCase):

    def test_gen_node_list(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1 = k._gen_node_list(repr1)
        result = "[(ADJ -> 'colorless', (0, 0)), (ADV -> 'furiously', (1, 1)), (N -> 'ideas', (0, 1)), (NP -> ADJ N, (0,)), (S -> NP VP, ()), (V -> 'sleep', (1, 0)), (VP -> V ADV, (1,))]"
        self.assertEqual(str(nodes1), result)

    def test_get_node_pair_list(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1 = k._gen_node_list(repr1)
        nodes2 = k._gen_node_list(repr2)
        node_list = k._get_node_pair_list(nodes1, nodes2)
        result = "[((0, 0), (0, 0), 0), ((1, 1), (1, 1), 0), ((0, 1), (0, 1), 0), ((1, 0), (1, 0), 0), ((0,), (0,), 2), ((1,), (1,), 2), ((), (), 2)]"
        self.assertEqual(str(node_list), result)

    def test_get_node_pair_list_cy(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1 = k._gen_node_list(repr1)
        nodes2 = k._gen_node_list(repr2)
        node_list = k._get_node_pair_list_cy(nodes1, nodes2)
        result = "[((0, 0), (0, 0), 0), ((1, 1), (1, 1), 0), ((0, 1), (0, 1), 0), ((1, 0), (1, 0), 0), ((0,), (0,), 2), ((1,), (1,), 2), ((), (), 2)]"
        self.assertEqual(str(node_list), result)


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

    def test_prof_K(self):
        X = np.array([['(S (NP ns) (VP v))'],
                      ['(S (NP n) (VP v))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        k = SST()
        target = np.zeros(shape=(len(X), len(X)))
        ITS = 2000
        start_time = datetime.datetime.now()
        for i in range(ITS):
            k.K(X, None, target)
        end_time = datetime.datetime.now()
        print target/ITS
        print end_time - start_time

if __name__ == "__main__":
    unittest.main()
