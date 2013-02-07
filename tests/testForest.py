#!/usr/bin/env python3
#encoding: utf-8

import unittest
import collections
import itertools
import copy
import forest
import model

class ForestOpsTestCase(unittest.TestCase):
    """
    Testing basic operations on forest, not strategy- or model-specific.
    Content of the nodes is irrelevant.
    """
    def setUp(self):
        # create a forest of one binary tree with self.size nodes.
        self.size = 8
        self.f = forest.Forest(set([self.createTree(self.size)]))
    
    def createTree(self, size):
        nodes = collections.deque([forest.Node() for i in range(size)])
        root = nodes.pop()
        
        def addPreds(succ, nodeQueue):
            succ.subtreeSize = len(nodeQueue) + 1
            if len(nodeQueue):
                succ.preds.add(nodeQueue.pop())
            if len(nodeQueue):
                succ.preds.add(nodeQueue.pop())
            for node in succ.preds:
                addPreds(node, nodeQueue)
            return
        
        addPreds(root, nodes)
        return root
    
    def testTraverse(self):
        """traverse must visit all nodes in a forest"""
        self.assertEqual(len(list(self.f.traverse())), self.size)
    
    def test_removeNode(self):
        """_removeNode must keep node's subtrees as new trees in the forest"""
        someRoot = list(self.f.roots)[0]
        expectedRootNum = len(self.f.roots) - 1 + len(someRoot.preds)
        expectedForestSize = len(list(self.f.traverse())) - 1
        
        self.f._removeNode(someRoot)
        
        self.assertEqual(len(self.f.roots), expectedRootNum)
        self.assertEqual(len(list(self.f.traverse())), expectedForestSize)
    

class PredicateOpsTestCase(unittest.TestCase):
    """
    Testing predicate logic operations on the forests. 
    Content of the nodes is relevant.
    """
    def setUp(self):
        self.fPairs = []
        
        root = forest.Node.factory(0, [('a',0)])
        root.subtreeSize = 4
        node = forest.Node.factory(2, [('c',0)], root)
        node.subtreeSize = 2
        node.preds.add(forest.Node.factory(3, [], node))
        root.preds = set([node, forest.Node.factory(1, [('b',0)], root)])
        f1 = forest.Forest(set([root, forest.Node.factory(4, [('d',0)])]))
        
        root = forest.Node.factory(5, [])
        root.subtreeSize = 5
        node = forest.Node.factory(0, [('a',0), ('e',0)], root)
        node.subtreeSize = 3
        node.preds = set([forest.Node.factory(1, [('b',0)], node),\
        forest.Node.factory(6, [], node)])
        root.preds = set([node, forest.Node.factory(4, [('d',0)], root)])
        f2 = forest.Forest(set([root, forest.Node.factory(3, [])]))     
        
        self.fPairs.append((forest.Forest(set()), forest.Forest(set())))
        self.fPairs.append((copy.deepcopy(f1), forest.Forest(set())))
        self.fPairs.append((copy.deepcopy(f1), copy.deepcopy(f1)))
        self.fPairs.append((copy.deepcopy(f1), copy.deepcopy(f2)))
        
        return
    
    def equivalentIn(self, node, forest, predicate):
        """Check if forest has a node with the same state and an equivalent
        set of rules. 
        Predicate is a subset (weak), equivalence or a superset (strong).
        """
        nodeRules = node.allRules()
        for nd in forest.traverse():
            ndRules = nd.allRules()
            if node.state == nd.state and predicate(nodeRules, ndRules):
                return True
        return False
    
    def testConjunctionSoundness(self):
        """Every node in the result must have an equivalent in one operand
        and a weak equivalent in the other operand.
        """
        for psi1, psi2 in self.fPairs:
            origPsi1 = copy.deepcopy(psi1)
            origPsi2 = copy.deepcopy(psi2)
            
            psi1.conjunction(psi2)
            
            exEq = lambda n1, n2: n1 == n2
            wEq = lambda n1, n2: n1 >= n2
            for node in psi1.traverse():
                exEqPsi1 = self.equivalentIn(node, origPsi1, exEq)
                exEqPsi2 = self.equivalentIn(node, origPsi2, exEq)
                wEqPsi1 = self.equivalentIn(node, origPsi1, wEq)
                wEqPsi2 = self.equivalentIn(node, origPsi2, wEq)
                
                self.assertTrue((exEqPsi1 and wEqPsi2) or 
                                (exEqPsi2 and wEqPsi1))
    
    def testConjunctionCompleteness(self):
        """For each operand and each its node that is not represented in
        the result there must be no weak equivalent in the other operand.
        """
        for psi1, psi2 in self.fPairs:
            origPsi1 = copy.deepcopy(psi1)
            origPsi2 = copy.deepcopy(psi2)
            
            psi1.conjunction(psi2)
            
            wEq = lambda n1, n2: n1 >= n2
            # check left operand
            for node in origPsi1.traverse():
                if not self.equivalentIn(node, psi1, wEq):
                    self.assertFalse(self.equivalentIn(node, origPsi2, wEq))
            # check right  operand
            for node in origPsi2.traverse():
                if not self.equivalentIn(node, psi1, wEq):
                    self.assertFalse(self.equivalentIn(node, origPsi1, wEq))
    
    def testDisjunctionSoundness(self):
        """For each node from the result there must be the same
        node in one of the operands.
        """
        for psi1, psi2 in self.fPairs:
            origPsi1 = copy.deepcopy(psi1)
            origPsi2 = copy.deepcopy(psi2)
            
            psi1.disjunction(psi2)
            
            exEq = lambda n1, n2: n1 == n2
            for node in psi1.traverse():
                self.assertTrue(self.equivalentIn(node, origPsi1, exEq) or
                                self.equivalentIn(node, origPsi2, exEq))
    
    def testDisjunctionCompleteness(self):
        """For each node from each operand there must be the same node in
        the result.
        """
        for psi1, psi2 in self.fPairs:
            origPsi1 = copy.deepcopy(psi1)
            origPsi2 = copy.deepcopy(psi2)
            
            psi1.disjunction(psi2)
            
            exEq = lambda n1, n2: n1 == n2
            # check left operand
            for node in origPsi1.traverse():
                self.assertTrue(self.equivalentIn(node, psi1, exEq))
            # check right  operand
            for node in origPsi2.traverse():
                self.assertTrue(self.equivalentIn(node, psi1, exEq))
    

class FactoriesTestCase(unittest.TestCase):
    """
    Testing forest factories for atomic predicates specifying global states.
    Content of nodes is model-specific.
    """
    def testLocalDisjunction(self):
        ioPairs = []
        
        locStates = [[0],[1],[2]]
        expStates = set([(0,1,2)])
        ioPairs.append((locStates, expStates))
        
        locStates = [range(3) for i in range(3)]
        expStates = set(itertools.product(range(3), repeat=3))
        ioPairs.append((locStates, expStates))
        
        locStates = [[0,1,2], [0], [1,2]]
        expStates = set([(0,0,1),
                        (0,0,2),
                        (1,0,1),
                        (1,0,2),
                        (2,0,1),
                        (2,0,2)])
        ioPairs.append((locStates, expStates))
        
        for (locStates, expStates) in ioPairs:
            rootStates = set([r.state for r in \
                          forest.Forest.localDisjunction(locStates).roots])
            self.assertEqual(rootStates, expStates)
    
    def testCombinationPredicate(self):
        orAgt = model.agt
        model.agt = 3
        orSt = model.st
        model.st = 2
        ioPairs = []
        
        args = (1, set([1]))
        output = set([  (1,0,0),
                        (1,0,1),
                        (1,1,0),
                        (1,1,1),
                        (0,1,0),
                        (0,1,1),
                        (0,0,1)])
        ioPairs.append((args, output))
        
        args = (2, set([0]))
        output = set([  (0,0,0),
                        (0,0,1),
                        (0,1,0),
                        (1,0,0)])
        ioPairs.append((args, output))
        
        args = (3, set([0,1]))
        output = set(itertools.product(range(2), repeat=3))
        ioPairs.append((args, output))
        
        for (args, output) in ioPairs:
            rootStates = set([r.state for r in \
                            forest.Forest.combinationPredicate(*args).roots])
            self.assertEqual(rootStates, output)
            
        model.agt = orAgt
        model.st = orSt
        
    

if __name__ == '__main__':
    unittest.main(verbosity=2)
