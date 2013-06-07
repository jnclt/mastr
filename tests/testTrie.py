#!/usr/bin/env python3
# encoding: utf-8
import pickle
import unittest

import trie
import utils


class TrieTestCase(unittest.TestCase):
    """
    Testing consistency of Trie
    """
    def setUp(self):
        """
        Load the serialized test tree and construct the trie. 
        """
        with open("./tests/testTree.bin", 'rb') as f:
            phi = pickle.Unpickler(f).load()
        with open("./tests/testPerceptions.bin", 'rb') as f:  
            utils.usedPerceptions = pickle.Unpickler(f).load()
        with open("./tests/testStates.bin", 'rb') as f:   
            utils.usedStates = pickle.Unpickler(f).load()
        
        self.t = trie.Trie(phi)
        self.t.strategy()
    
    # def testTraverse(self):
    #     """traverse must visit all nodes in the trie"""
    #     # TODO deterministic check
    #     pass
    #     # self.assertEqual(len(list(self.t.traverse())), 150)
    
    def testDeltas(self):
        """Descendants must add deltas to ancestors.
        """
        gen = self.t.traverse()
        next(gen) # skip root
        for trieNode in gen:
            self.assertGreaterEqual(trieNode.states, trieNode.parent.states)
            self.assertEqual(trieNode.states,
            trieNode.parent.states | trieNode.delta)

if __name__ == '__main__':
    unittest.main(verbosity=2)