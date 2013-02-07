#!/usr/bin/env python3
#encoding: utf-8

import utils
import model
import trie
import copy
from collections import deque
from itertools import product, combinations


class Node:
    """
    One node of the forest
    """
    
    def __init__(self, node=None):
        """ 
        Copy the state and the rules of the node.
        """
        """ system state = list of |agt| individual states (ints < model.st)
        """
        if node:
            self.state = utils.usedStates[node.state]
        else:
            self.state = ()
        """set of rules
        (perception = k+1-tuple of ints < model.st, action = int < model.act)
        """
        if node:
            self.rules = node.rules.copy()
        else:
            self.rules = set([])
        """succeding node (Node)"""
        self.succ = None
        """set of preceding nodes (Node)"""
        self.preds = set([])
        """size of the entire subtree rooted at the node (including the node)
        """
        self.subtreeSize = 1
        """helper flag for marking nodes to filter for various purposes
        """
        self.included = False
        return
    
    def __repr__(self):
        return '(%s:%d, %s)' % (self.state, self.subtreeSize, 
        list(self.rules))
    
    @classmethod
    def factory(cls, state, rules, succ=None):
        """
        Return new Node representing tuple 'state' and rules from
        iterable 'rules', having successor Node 'succ'.
        """
        n = cls()
        n.state = utils.getRefState(state)
        n.rules = set([utils.getRefRule(rule) for rule in rules])
        n.succ = succ
        return n
    
    def equals(self, node):
        """ 
        Return True if self and node represent the same state and rules.
        """
        return self.state == node.state and self.rules == node.rules
    
    @staticmethod
    def conflict(rules1, rules2):
        """
        Return True if a rule from rules1 is conflicting with a rule from
        rules2.
        Return False otherwise.
        """
        for rule1 in rules1:
            if rule1[0] in rules2 and rule1[1] != rules2[rule1[0]]:
                return True
        return False
    
    def __deepcopy__(self, memo):
        """
        Return the copy of the entire subtree rooted at self.
        """
        cp = Node(self)
        # recursion for all preds
        for pred in self.preds:
            cp.preds.add(copy.deepcopy(pred, memo))
        # set the successor of all preds to copy
        for pred in cp.preds:
            pred.succ = cp
        # set the subtreeSize
        cp.subtreeSize = self.subtreeSize
        # set the 'included' flag
        cp.included = self.included
        
        return cp
    
    def root(self):
        """
        Return node's root.
        """
        root = self
        while root.succ:
            root = root.succ
        return root
    
    def changeSubtreeSize(self, delta):
        """
        Change the subtreeSize of self with delta and propagate change to
        all successors up to the root.
        """
        if delta == 0:
            return
        self.subtreeSize += delta
        if self.succ:
            self.succ.changeSubtreeSize(delta)
        return
    
    def allRules(self):
        """
        Return the union of all rules on the way from self to the root.
        """
        if not self.succ:
            return self.rules
        return self.rules | self.succ.allRules()
    
    def sortPredecessors(self):
        """
        Sort the set of predecessors into a list according to the sizes of
        their subtrees.
        Recursively for the entire subtree rooted at self.
        Changes preds from set to list!
        """
        self.preds = sorted(self.preds, key=lambda pred: pred.subtreeSize,
        reverse=True)
        for pred in self.preds:
            pred.sortPredecessors()
        return
    
    def cpPrune(self, rules):
        """
        Return a copy of the subtree rooted at self and pruned with rules.
        """
        return self._cpUpdate(rules, False)
    
    def cpPropagate(self, rules):
        """
        Return a copy of the subtree rooted at self where the nodes'
        rules are extended with the 'rules' when compatible and the nodes
        are removed when incompatible.
        """
        return self._cpUpdate(rules, True)
    
    def _cpUpdate(self, rules, propagate=False):
        """
        Return a copy of the subtree rooted at self and pruned with
        'rules'. If 'propagate' is True then the nodes in the copy are
        extended with 'rules'.
        """
        # check if there is a conflict between self.rules and rules 
        if Node.conflict(self.rules, rules):
            return None
        # otherwise, recursively continue with the subtree
        copy = Node(self)
        if propagate:
            copy.rules |= rules
        for pred in self.preds:
            newPred = pred._cpUpdate(rules, propagate)
            if newPred:
                newPred.succ = copy
                copy.preds.add(newPred)
                copy.subtreeSize += newPred.subtreeSize
                
        return copy
    
    def preimage(self, strategyRules):
        """
        Return the set of all possible predecessors of itself that are
        compatible with given strategy
        """
        # 1. for each individual state q find the rules with outcome q
        rules = {}
        for q in self.state:
            if q not in rules:
                rules[q] = utils.wTran_1(q) # set of rules leading to q
                if not len(rules[q]):
                    # there are no predecessors of this node
                    return set([])
                    
        # 2. filter the rules with those from the given strategy
        doubles = set([]) # rules already present in chunk strategies
        for strRule in strategyRules:
            for q in rules.keys():
                toRemove = set([])
                for rule in rules[q]:
                    if strRule[0] == rule[0]:
                        if strRule[1] != rule[1]:
                            # conflict
                            toRemove.add(rule)
                        else:
                            # dopplegaenger
                            doubles.add(rule)
                rules[q] -= toRemove
        
        # 3. compute predecessors from the rules
        preds = self._findPreds(rules, [])
        
        # 4. remove double rules and append the new nodes
        for pred in preds:
            pred.rules -= doubles
            pred.succ = self
        
        return preds
    
    def _findPreds(self, rules, vector):
        """
        Return the set of all preceding Nodes with system states starting with
        'vector' from which self is reached by executing rules from
        'rules' by the respective agents:
        vector - list of pairs (individual state, rule) for the first x agents
        rules - dictionary of pairs (individual state 'q', set of rules 'r')
        where q is a state from self and 'r' are some rules with outcome q
        """
        # next agent to resolve
        a = len(vector)
        if a == model.agt:
            # all agents resolved -> construct new node
            node = Node()
            Q = [rule[0][0] for rule in vector]
            node.state = utils.getRefState(tuple(Q))
            node.rules = set(vector)
            return set([node])
        
        # for each rule of agent a, try to complete a strategy containing
        # this rule
        preds = set([])
        for rule in rules[self.state[a]]:
            conflicts = 0
            # for each neighbor n
            for i in range(model.k):
                n = model.neig(a, i)
                # conflict if a's perception does not reflect n's state
                if -1 < n < a \
                and (rule[0][i + 1] != model.manif(vector[n][0][0]) \
                or vector[n][0][model.neig_1(n, a) + 1] != \
                model.manif(rule[0][0])):
                    conflicts += 1
                # conflict if a's rule collides with the fixed rules
                for fixedRule in vector:
                    if fixedRule[0] == rule[0] \
                    and fixedRule[1] != rule[1]:
                        conflicts += 1
            
            if conflicts == 0:
                vector.append(rule)
                preds |= self._findPreds(rules, vector)
                vector.pop()
        return preds
    
    def singleExport(self, fname, format="dot"):
        """
        Export the subtree rooted at self in dot format into a file with
        fname.
        """
        with open(fname, 'w') as f:
            f.truncate(0)
            f.write(utils.graphHeader[format])
            self.export(f, format)
            f.write(utils.graphFooter[format])
        return
    
    def export(self, f, format):
        """
        Export the subtree rooted at self as a part of a graph in 'format'
        into open file 'f'.
        """
        self._traverseNode(f, format)
        self._traverseEdge(f, format)
        return
    
    def _traverseNode(self, f, format):
        """
        Write istelf in 'format' into file 'f'.
        """
        nodeStr = ""
        if format == "GraphML":
            if __debug__:
                prefixStr = '\t<node id="{0}">\n'.format(id(self))
                colorStr = '\t\t<data key="c">{0}</data>\n'.format("grey" if
                self.included else "white")
                labelStr = '\t\t<data key="l">{0}:{1}\n'.format(self.state,
                self.subtreeSize)
                ruleStr = ''.join(['\t\t\t{0} -> {1}\n'.format(rule[0],
                rule[1]) for rule in self.rules])
                suffixStr = '\t\t</data>\n\t</node>\n'
                nodeStr = ''.join([prefixStr, colorStr, labelStr, ruleStr,
                suffixStr])
            else:
                nodeStr = '\t<node id="{0}">\n\t\t<data \
                key="l">{1}</data>\n\t</node>\n'.format(id(self), self.state)
        else: # dot format
            if __debug__:
                nodeStr = '{0} [label="{1}:{2}"];\n'.format(id(self),
                self.state, self.rules)
            else:
                nodeStr = '{0} [label="{1}"];\n'.format(id(self), self.state)
        f.write(nodeStr)
        
        for pred in self.preds:
            pred._traverseNode(f, format)
    
    def _traverseEdge(self, f, format):
        """
        Write all edges to its predecessors in format 'format' into file 'f'.
        """
        if self.succ:
            edgeStr = ""
            if format == "GraphML":
                edgeStr = '<edge source="{0}" target="{1}"/>\n'.format(
                id(self), id(self.succ))
            else: # dot format
                edgeStr = '"{0}"->"{1}\";\n'.format(id(self), id(self.succ))
            f.write(edgeStr)
            
        for pred in self.preds:
            pred._traverseEdge(f, format)
    

class Forest:
    """
    Forest representing the union of all strategies for a formula
    """
    
    def __init__(self, roots):
        """set of root nodes (Node)"""
        self.roots = roots
        if __debug__:
            self.depth = 0
        return
    
    def export(self, fname, format="GraphML"):
        """
        Export the subtree rooted at self in 'dot' or 'GraphML' format
        into a file with 'fname'.
        """
        with open(fname, 'w') as f:
            f.truncate(0)
            f.write(utils.graphHeader[format])
            for r in self.roots:
                r.export(f, format)
            f.write(utils.graphFooter[format])
            f.close()
        return
    
    def traverse(self, bfs=False):
        """
        Generator for traversing self.
        Depth first by default, breadth first if 'bfs'
        """
        dq = deque(self.roots)
        while dq:
            node = dq.popleft()
            if bfs:
                dq.extend(node.preds)
            else:
                dq.extendleft(node.preds)
            yield node
    
    def _removeNode(self, node):
        """
        Remove the node 'node' from self.
        """ 
        if node in self.roots:
            self.roots.remove(node)
        if node.succ:
            node.succ.preds.remove(node)
            node.succ.changeSubtreeSize( - node.subtreeSize)
        for pred in node.preds:
            pred.rules = pred.allRules()
            pred.succ = None
        self.roots |= node.preds
        return
    
    def _filterOut(self, predicate):
        """
        Remove each node that satisfies predicate. 'predicate' must be
        a function accepting Node and returning Boolean.
        """
        toRemove = [node for node in self.traverse() if predicate(node)]
        for node in toRemove:
            self._removeNode(node)
        return  
    
    @classmethod
    def true(cls):
        """
        Factory for the forest of nodes representing all system states.
        No rules are assigned.
        """
        roots = [Node.factory(state, []) for state in \
        product(range(model.st), repeat=model.agt)]
        return cls(set(roots))
    
    @classmethod
    def localDisjunction(cls, locStates):
        """
        Factory for the forest of nodes representing all system states 
        composed of 'locStates'.
        No rules are assigned.
        locStates is a list of |model.agt| iterables over local states.
        locStates[i] is an iterable over all local states that agent i 
        may be at.
        """
        # TODO move the check before the outside call of localDisjunction
        # assert len(locStates) == model.agt,\
        # "Local states for {0} instead of {1} agts specified.".format(\
        # len(locStates), model.agt)
        # for (agt, agtStates) in enumerate(locStates):
        # assert len(agtStates) >= 1, "No states for agt {0}".format(agt)
        roots = [Node.factory(state, []) for state in \
        cls._globalSuffixes(locStates, len(locStates))]
        return cls(set(roots))
    
    @classmethod
    def _globalSuffixes(cls, locStates, length):
        """
        Generate list of the suffixes  of given 'length' of all global states
        composed of local states specified in locStates.
        """
        if length == 0:
            return [()]
        
        newSuffixes = []
        oldSuffixes = cls._globalSuffixes(locStates, length-1)
        for agtState in locStates[len(locStates) - length]:
            for suffix in oldSuffixes:
                newSuffixes.append((agtState,) + suffix)
        return newSuffixes
    
    @classmethod
    def combinationPredicate(cls, agtNum, agtStates):
        """
        Factory of the forest for all system states, where |agtNum| agents
        have local states from the set of agtStates.
        """
        # TODO check args for correctness before calling
        # 0 < agNum <= model.agts, emptyset < agtStates <= model.st
        rootStates = []
        # for every |agtNum|-tuple of agents
        for comb in combinations(range(model.agt), agtNum):
            locStates = [None for agt in range(model.agt)]
            # specify the set of locals states for each of the agents
            for agt in comb:
                locStates[agt] = agtStates
            # set all local states as possible for the agents out of 
            # the tuple
            for (i, item) in enumerate(locStates):
                if item is None:
                    locStates[i] = range(model.st)
            # generate all nodes for local disjunction 
            rootStates.extend(cls._globalSuffixes(locStates, model.agt))
        
        roots = [Node.factory(state, []) for state in set(rootStates)]
        return cls(set(roots))
    
    def conjunction(self, f2):
        """
        Remove from both forests (self and f2) every node that has no weak
        equivalent in the other forest. By weak equivalent of a node N is
        meant a node representing the same state and a strategy that is
        a subset of N's strategy. Join the remnants of the two forests and
        assign them to self.
        """
        
        # 1. clean up the first forest
        self._filterOut(lambda n: not f2._hasWeakEquivalent(n))
        # 2. clean up the second forest
        f2._filterOut(lambda n: not n.included)
        # reset 'included' flags
        for node in f2.traverse():
            node.included = False
        # 3. join the remnants
        self.roots |= f2.roots
        return self
    
    def _hasWeakEquivalent(self, node):
        """
        Helper for Conjunction. 
        Return False if there is no weak equivalent of node in self, return
        True otherwise. Mark every strong equivalent of node in self with
        'included' flag. By strong equivalent of a node N is meant a node
        representing the same state and a strategy that is a strict superset
        of N's strategy.
        """
        
        equivalent = False
        nodeRules = node.allRules()
        for n in self.traverse():
            if n.state == node.state:
                nRules = n.allRules()
                if nRules > nodeRules:
                    n.included = True
                elif nRules <= nodeRules:
                    equivalent = True
        return equivalent
    
    def disjunction(self, f2):
        """
        Merge the two forests (self and f2) into self.
        """
        # TODO check for duplicates
        self.roots |= f2.roots
        return self
    
    def next(self):
        """
        Transform the forest (for psi) into the forest for X(psi).
        """
        # traverse and extend the original forest (DFS)
        for root in self.roots:
            self._recNext(root, set([]))
        # remove old roots and designate their predecessors as new roots
        newRoots = set([])
        for root in self.roots:
            for pred in root.preds:
                pred.succ = None
            newRoots |= root.preds
        self.roots = newRoots
        return self
    
    def _recNext(self, node, strategyRules):
        """ 
        Recursive subroutine for Next.
        Recursively append all possible predecessors compatible with
        strategyRules to the subtree of itself rooted at node.
        """
        # extend strategyRules with those from node
        strategyRules |= node.rules
        # run recursively on all 'original' predecessors
        for pred in node.preds:
            self._recNext(pred, strategyRules)
        # append the preimage of the node
        preimage = node.preimage(strategyRules)
        singles = set(preimage)
        for pred in node.preds:
            for prei in preimage:
                if pred.equals(prei):
                    singles.remove(prei)
        node.preds |= singles
        node.changeSubtreeSize(len(singles))
        # restore the original strategy
        strategyRules -= node.rules
        return  
    
    def always(self):
        """
        Return forest for G(psi).
        """
        newRoots = set()
        for root in self.roots:
            newRoots |= self._cycle(root, root.rules)
        return Forest(newRoots)
    
    def _cycle(self, node, strategyRules):
        """
        Recursive subroutine for Always
        Extend 'node' having 'strategyRules' with the (pruned)
        trees from self, until a cycle is formed (roots correspond) or
        there is nothing to attach.
        Return the roots of trees with cycle closed at or under the node,
        which are extended to maximum. 
        """
        newRoots = set()
        knot = node.root()
        # find predecessors
        preimage = node.preimage(strategyRules)
        for pred in preimage:
            # check for the cycle closed with knot
            if pred.state == knot.state and\
                not Node.conflict(knot.rules, pred.rules):
                # cycle closed, get the pruned copy of tree
                newTree = knot.cpPropagate(pred.rules)
                # extend the tree with the cycle as much as possible
                f = Forest(set([newTree]))
                f = f.until(copy.deepcopy(self))
                newRoots |= f.roots
            else:
                # attach the trees from self to node
                delta = 0
                roots = [root for root in self.roots if\
                            root.state == pred.state and\
                            not Node.conflict(root.rules, pred.rules)]
                for root in roots:
                    newTree = root.cpPropagate(pred.rules)
                    if newTree:
                        node.preds.add(newTree)
                        delta += 1
                node.changeSubtreeSize(delta)
                # search for cycle recursively
                for pred in node.preds:
                    newRoots |= self._cycle(pred, strategyRules | pred.rules)
        return newRoots
    
    def until(self, f1):
        """
        Given the forest f1 for psi1, it transforms the forest (for psi2) into
        the forest for (psi1)U(psi2)
        """
        # 1. Treat the roots of psi2 seperately, in case that they have
        # equivalents among the roots for psi1. 
        # Copy the tree into psi2 if the root from psi1 is at least the same
        # as the equivalent root from psi2.
        rootsToAdd = set([])
        for rootPsi2 in self.roots:
            for rootPsi1 in f1.roots:
                if rootPsi1.state == rootPsi2.state:
                    if set(rootPsi1.rules) >= set(rootPsi2.rules):
                        rootsToAdd.add(copy.deepcopy(rootPsi1))
        self.roots |= rootsToAdd
        
        # 2. traverse psi2 (DFS) and merge into it pruned copies of trees from
        # psi1
        for root in self.roots:
            self._appendPsi1(root, set([]), root, f1)
            
        return self
    
    def _appendPsi1(self, node, strategyRules, root, f1):
        """
        Recursive subroutine for Until.
        Extend the subtree of itself rooted at 'node', which is a subtree of
        a tree rooted at 'root', with the subforest of f1 that is compatible
        with strategyRules.
        """
        if __debug__:
            self.depth += 1
            print("append-depth:", self.depth)
            if self.depth == 50:
                self.export(utils.outputPath() + "tree.graphml", "GraphML")
                print("Depth reached 50 => exported")
        # extend strategyRules with those from node
        strategyRules |= node.rules
        # find the preimage of the node
        preimage = node.preimage(strategyRules)
        # for each predecessor from the preimage check if it corresponds to 
        # a root from psi1-forest
        for pred in preimage:
            for f1root in f1.roots:
                # print pred.state, 'vs.', f1root.state
                if pred.state == f1root.state:
                    # if yes, check that it doesn't form a cycle (to prevent
                    # repetitive unfolding)
                    if f1root.state != root.state:
                        # if not, append a pruned copy of the tree rooted at
                        # the root to the node in psi2:
                    
                        # 1. create the pruned copy
                        extendedRules = strategyRules | pred.rules
                        newRoot = f1root.cpPrune(extendedRules)
                        if not newRoot:
                            continue
                        newRoot.rules |= pred.rules
                        newRoot.rules -= strategyRules
                        # 2. merge the pruned copy into psi2
                        self._merge(newRoot, node)
                    
        # recursion for all *updated* predecessors of the node in psi2
        for pred in node.preds:
            self._appendPsi1(pred, strategyRules, root, f1)
        
        # restore the original strategyRules
        strategyRules -= node.rules
        
        if __debug__:
            self.depth -= 1
        
        return
    
    def _merge(self, node, parent):
        """
        Recursive subroutine for Until.
        Merge the tree rooted at the node into the subtree of self rooted at
        the parent.
        Simply append to the parrent if node is not equal to any of parent's
        predecessors, recursively merge otherwise.
        """
        # 1. check if the node is already represented by some predecessor of
        # the parent
        present = False
        for pred in parent.preds:
            if pred.equals(node):
                # 2. if yes, recursively merge the tree rooted at the node 
                # into the parent's subtree 
                present = True
                for nodePred in node.preds:
                    self._merge(nodePred, pred)
                break
                
        # 3.otherwise, append the node to the parent
        if not present:
            parent.preds.add(node)
            node.succ = parent
            parent.changeSubtreeSize(node.subtreeSize)
            
        return
    
    def totalStrategy(self):
        """Return a single strategy defined for all states included in 
        the forest, if there is one. Return 'None' otherwise.
        """
        # 1. Check that the forest at least contains all states
        
        # At least every state must have been created so far
        if len(utils.usedStates) != pow(model.st, model.agt):
            if __debug__:
                print('utils.usedStates:', len(utils.usedStates))
            return None
        # Traverse the forest while checking against the set of all states
        noncoveredStates = set(product(range(model.st), repeat=model.agt))
        for node in self.traverse():
            noncoveredStates.discard(node.state)
            
        if noncoveredStates:
            if __debug__:
                print('non-covered states:', len(noncoveredStates))
                print(noncoveredStates)
            return None
        
        # 2. Try to construct the strategy covering all states        
        strTrie = trie.Trie(self)
        # Build the trie and return the strategy if there is one
        strategy = strTrie.strategy()
        if __debug__:
            strTrie.export(utils.outputPath() + "trie.graphml")
        # TODO move into a unittest for Trie
        if __debug__:
            gen = strTrie.traverse()
            next(gen) # skip root
            for trieNode in gen: 
                assert trieNode.states >= trieNode.parent.states, "trie degen"
                assert trieNode.states == \
                trieNode.parent.states | trieNode.delta, "delta degen"
        return strategy
    