#!/usr/bin/env python3
#encoding: utf-8

import forest


def phi():
    """ Create the forest for model formula.
    """
    # phi = majority_0 & FG 0000 | majority_1 & FG 1111 | equal_0_1
    # model consensus 4
    r1 = forest.Node.factory((0,0,0,0), [((0,0,0), 0)])
    r2 = forest.Node.factory((1,1,1,1), [((1,1,1), 0)])
    
    # model consensus 5
    # r1 = forest.Node.factory((0,0,0,0,0), [((0,0,0), 0)])
    # r2 = forest.Node.factory((1,1,1,1,1), [((1,1,1), 0)])
    
    # model consensus 6
    # r1 = forest.Node.factory((0,0,0,0,0,0), [((0,0,0), 0)])
    # r2 = forest.Node.factory((1,1,1,1,1,1), [((1,1,1), 0)])
    
    # model consensus 7
    # r1 = forest.Node.factory((0,0,0,0,0,0,0), [((0,0,0), 0)])
    # r2 = forest.Node.factory((1,1,1,1,1,1,1), [((1,1,1), 0)])
    
    psi2a = forest.Forest(set([r1]))
    psi2b = forest.Forest(set([r2]))
    
    psi1a = forest.Forest.true()
    psi1b = forest.Forest.true()
    
    # majority 0
    psi3a = forest.Forest.combinationPredicate(3, set([0]))
    
    # majority 1
    psi3b = forest.Forest.combinationPredicate(3, set([1]))
    
    # operands for equality
    psi4a = forest.Forest.combinationPredicate(2, set([1]))
    psi4b = forest.Forest.combinationPredicate(2, set([0]))
    
    psi2a = psi2a.until(psi1a).conjunction(psi3a)
    psi2b = psi2b.until(psi1b).conjunction(psi3b)
    psi4 = psi4a.conjunction(psi4b)
    psi2a = psi2a.disjunction(psi2b)
    psi2a = psi2a.disjunction(psi4)
    
    phi = psi2a
    return phi
