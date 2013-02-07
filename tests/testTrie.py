#!/usr/bin/env python3
# encoding: utf-8

import unittest
import pickle

class TrieTestCase(unittest.TestCase):
    """
    Testing operations on Trie
    """
    # def setUp(self):
    #     """
    #     Load the serialized test tree and construct the trie. 
    #     """
    #     with open("./testTree.bin", 'rb') as f:
    #         phi = pickle.Unpickler(f).load()
    #     with open("./testPerceptions.bin", 'rb') as f:  
    #         utils.usedPerceptions = pickle.Unpickler(f).load()
    #     with open("./testStates.bin", 'rb') as f:   
    #         utils.usedStates = pickle.Unpickler(f).load()
    pass

if __name__ == '__main__':
    unittest.main(verbosity=2)