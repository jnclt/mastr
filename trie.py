#!/usr/bin/env python3
#encoding: utf-8

import model
import utils
import strategy
import forest
from collections import deque
from itertools import product

class TrieNode:
    def __init__(self):
        """set of states covered by the rules on the way to the root"""
        self.states = set([])
        """subset of self.states - those that are not included in
        parent.states"""
        self.delta = set([])
        """level on which the node resides in the trie (root has level 0)"""
        self.level = 0
        """action from range(model.act) assigned to the perception at
        self.level"""
        self.action = 0
        """parent TrieNode"""
        self.parent = None
        """model.act-long list of the children"""
        self.children = [None] * model.act
    
    def export(self, f, levels, format):
        """
        Export the subtrie rooted at self as a part of a dot graph into
        open file f.
        """
        self._traverseNode(f, format)
        self._traverseEdge(f, levels, format)
        return
    
    def _traverseNode(self, f, format):
        """
        Write istelf as a dot command into file f.
        """
        nodeStr = ""
        if format == "GraphML":
            if __debug__:
                prefixStr = '\t<node id="{0}">\n'.format(id(self))
                labelStr = '\t\t<data key="l">\n'
                statesStr = ''.join(['\t\t\t{0}\n'.format(state) for state in 
                self.states])
                suffixStr = '\t\t</data>\n\t</node>\n'
                nodeStr = ''.join([prefixStr, labelStr, statesStr, suffixStr])
            else:
                nodeStr = '\t<node id="{0}">\n\t\t<data key="l">\
                {1}</data>\n\t</node>\n'.format(id(self), self.delta)
        else: # dot format
            if __debug__:
                nodeStr = '{0} [label="{1}"];\n'.format(id(self), self.delta)
            else:
                nodeStr = '{0} [label="{1}"];\n'.format(id(self), self.delta)
        f.write(nodeStr)
        
        for child in self.children:
            if child:
                child._traverseNode(f, format)
    
    def _traverseEdge(self, f, levels, format):
        """
        Write all edges to its predecessors as dot commands into file f.
        """
        for i in range(len(self.children)):
            child = self.children[i]
            if child:
                edgeStr = ""
                if format == "GraphML":
                    edgeStr = '\t<edge source="{0}" target="{1}">\n\t\t\<data\
                    key="e">{2}:{3}</data>\n\t</edge>\n'.format(id(self),
                    id(child), levels[self.level], i)
                else: # dot format
                    edgeStr = '"{0}"->"{1}\";\n'.format(id(self), id(child))
                f.write(edgeStr)
        for child in self.children:
            if child:
                child._traverseEdge(f, levels, format)
        return
    


class Trie:
    """
    Prefix tree (trie) for finding a complete strategy. Constructed from
    the forest for a formula. An edge represents a rule (perception, action),
    all edges on the same level represent the same rule. The edges leading 
    to one node are sorted by the action assigned to the perception.
    Nodes contain all system states for which the strategy composed of
    the rules on the way to the root works.
    As soon as a node containing all states to be covered is added, we are
    finished - the strategy composed of the rules on the branch is the total
    strategy.
    Following the ordering of the preds in the forest and putting the most 
    used rules high in the trie, we should be keeping the size of the trie 
    close to minimal.  
    """
    def __init__(self, forest):
        """mapping of rule perceptions to the levels"""
        self.levels = [()] * len(utils.usedPerceptions)
        """set of all levels in trie"""
        self.levelSet = set([])
        """current highest level of the trie"""
        self.depth = 0
        """root of the trie (TrieNode)"""
        self.root = TrieNode()
        """forest with the strategy"""
        self.forest = forest
        """set of states that must be covered by one node"""
        self.statesToCover = set([])
    
    def strategy(self):
        """
        Build trie until a leaf containing all system states is created.
        Return the rules on the branch from the node to the root.
        """
        # 1. Create the set of all system states
        self.statesToCover = set(product(range(model.st), repeat=model.agt))
        
        # 2. Sort the predecessors in each node according to the sizes of 
        # their subtrees. node.preds becomes a sorted list instead of a set!
        # As a consequence, rules covering most states get high in the trie.
        
        # Add the common root
        cRoot = forest.Node()
        cRoot.preds = self.forest.roots
        self.forest.roots = set([cRoot])
        # Sort
        cRoot.sortPredecessors()
        
        # 3. Traverse the forest 
        # (BFS to take the advantage of the ordering of the preds)
        # and add each node to the trie
        for node in self.forest.traverse(bfs=True):
            strat = self.addNode(node)
            if strat:
                return strat
        
        return None
    
    def addNode(self, node):
        """
        Extend trie with new rules from the node.
        """
        for rule in node.rules:
            if rule[0] not in self.levelSet:
                self.levels[self.depth] = rule[0]
                self.levelSet.add(rule[0])
                self.depth += 1
        
        # disseminate node's state in trie
        # TODO improve - rely on the fact that the previously added node
        # was probably this node's sibling -> no need to start all the way
        # from the trie's root.
        # Get together all rules from the node to the tree's root:
        nodeStr = node.allRules().copy() 
        nodeStr = dict(nodeStr)
        # find the proper nodes in the trie to add the node.state
        strNode = self._propagateNode(self.root, nodeStr, node.state)
        # check if the node covering all states was added
        if strNode:
            # backtrack to root and put the rules on the way together.
            strat = strategy.Strategy()
            strat.domain = strNode.states
            while strNode.parent:
                strat.rules[self.levels[strNode.level-1]] = strNode.action
                strNode = strNode.parent
            return strat
            
        # states couldn't be covered
        return None
            
    def _propagateNode(self, trieNode, nodeStr, nodeState):
        """
        Add nodeState to trieNode if nodeStr is epmty.
        Otherwise, propagate nodeState deeper in the trie while following 
        the rules from nodeStr.
        Return trieNode if it covers all states to cover, None otherwise. 
        """
        # check if we arrived at the correct node
        if not nodeStr:
            if nodeState:
                # update the set of covered states in the node and all its
                # descendants
                trieNode.delta.add(nodeState)
                for node in self.traverse(trieNode):
                    node.states.add(nodeState)
                    if node.states == self.statesToCover:
                        return node
            return None
            
        currPerception = self.levels[trieNode.level]
        if currPerception not in nodeStr:
            # propagate in all children
            for i in range(len(trieNode.children)):
                if not trieNode.children[i]:
                    newNode = TrieNode()
                    newNode.level = trieNode.level + 1
                    newNode.action = i
                    newNode.states = trieNode.states.copy()
                    newNode.parent = trieNode
                    trieNode.children[i] = newNode
                node = self._propagateNode(trieNode.children[i],
                                          nodeStr, nodeState)
                if node:
                    return node
        else:
            # propagate in the corresponding child
            # consume the rule
            currAction = nodeStr.pop(currPerception)
            if not trieNode.children[currAction]:
                newNode = TrieNode()
                newNode.level = trieNode.level + 1
                newNode.action = currAction
                newNode.states = trieNode.states.copy()
                newNode.parent = trieNode
                trieNode.children[currAction] = newNode
            node = self._propagateNode(trieNode.children[currAction],
                                      nodeStr, nodeState)
            # finish if all states are covered
            if node:
                return node
            # put the rule back
            nodeStr[currPerception] = currAction
        
        return None
    
    def export(self, tname, format="GraphML"):
        """
        Export the trie in 'dot' or 'GraphML' format into 'tname' file.
        """
        with open(tname, 'w') as t:
            t.truncate(0)
            t.write(utils.graphHeader[format])
            self.root.export(t, self.levels, format)
            t.write(utils.graphFooter[format])
        return
    
    def traverse(self, root=None):
        """
        Generator for traversing (sub)trie rooted at 'root' 
        (depth first).
        """
        if not root:
            root = self.root
        dq = deque([root])
        while dq:
            node = dq.popleft()
            dq.extendleft([c for c in node.children if c])
            yield node

