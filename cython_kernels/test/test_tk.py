

import unittest
import nltk
from cython_kernels.tk.sst import SubsetTreeKernel as SST

class SSTTests(unittest.TestCase):

    def test_parse(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1 = k._gen_node_list(repr1)
        result = "[(ADJ -> 'colorless', (0, 0)), (ADV -> 'furiously', (1, 1)), (N -> 'ideas', (0, 1)), (NP -> ADJ N, (0,)), (S -> NP VP, ()), (V -> 'sleep', (1, 0)), (VP -> V ADV, (1,))]"
        self.assertEqual(str(nodes1), result)
        #print nodes1


if __name__ == "__main__":
    unittest.main()
