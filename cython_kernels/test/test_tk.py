

import unittest
import nltk
from cython_kernels.tk.sst import SubsetTreeKernel as SST
import datetime

class SSTTests(unittest.TestCase):

    def test_gen_node_list(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1 = k._gen_node_list(repr1)
        result = "[(ADJ -> 'colorless', (0, 0)), (ADV -> 'furiously', (1, 1)), (N -> 'ideas', (0, 1)), (NP -> ADJ N, (0,)), (S -> NP VP, ()), (V -> 'sleep', (1, 0)), (VP -> V ADV, (1,))]"
        self.assertEqual(str(nodes1), result)
        #print nodes1

    def test_get_node_pair_list(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1 = k._gen_node_list(repr1)
        nodes2 = k._gen_node_list(repr2)
        node_list = k._get_node_pair_list(nodes1, nodes2)
        result = "[((0, 0), (0, 0), 0), ((1, 1), (1, 1), 0), ((0, 1), (0, 1), 0), ((1, 0), (1, 0), 0), ((0,), (0,), 2), ((1,), (1,), 2), ((), (), 2)]"
        self.assertEqual(str(node_list), result)
        #print node_list


class SSTProfilingTests(unittest.TestCase):

    def test_prof_gen_node_pair_list(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1 = k._gen_node_list(repr1)
        nodes2 = k._gen_node_list(repr2)
        start_time = datetime.datetime.now()
        for i in range(100000):
            node_list = k._get_node_pair_list(nodes1, nodes2)
        end_time = datetime.datetime.now()
        print end_time - start_time

if __name__ == "__main__":
    unittest.main()
